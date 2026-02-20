import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from detectron2.data.datasets import register_coco_instances


def register_chess_datasets():
    from detectron2.data import DatasetCatalog
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dino"))

    if "chess_train" not in DatasetCatalog.list():
        train_json = os.path.join(data_root, "train/train/_annotations.coco.json")
        train_images = os.path.join(data_root, "train/train")
        if os.path.exists(train_json):
            register_coco_instances("chess_train", {}, train_json, train_images)
            print(f"Registered chess_train dataset")

    if "chess_val" not in DatasetCatalog.list():
        val_json = os.path.join(data_root, "val/valid/_annotations.coco.json")
        val_images = os.path.join(data_root, "val/valid")
        if os.path.exists(val_json):
            register_coco_instances("chess_val", {}, val_json, val_images)
            print(f"Registered chess_val dataset")


register_chess_datasets()

logger = logging.getLogger("detrex")


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


class Trainer(SimpleTrainer):

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler
                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        self.amp = amp
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


class BestCheckpointHook(hooks.HookBase):

    def __init__(self, eval_period, checkpointer, val_ann_path, val_img_dir,
                 confidence_threshold=0.3, iou_threshold=0.5):
        self._period = eval_period
        self._checkpointer = checkpointer
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._best_acc = -1.0
        self._val_anns, self._val_info = self._load_coco(val_ann_path, val_img_dir)

    @staticmethod
    def _load_coco(ann_path, img_dir):
        import json
        from collections import defaultdict
        with open(ann_path) as f:
            coco = json.load(f)
        cat_map = {c['id']: c['name'] for c in coco['categories']}
        anns = defaultdict(list)
        for a in coco['annotations']:
            x, y, w, h = a['bbox']
            anns[a['image_id']].append({
                'box': [x, y, x + w, y + h],
                'class': cat_map[a['category_id']]
            })
        info = {img['id']: os.path.join(img_dir, img['file_name']) for img in coco['images']}
        return anns, info

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def _compute_cls_accuracy(self, model):
        import cv2
        import numpy as np
        correct, total = 0, 0

        for img_id, annotations in self._val_anns.items():
            if img_id not in self._val_info:
                continue
            image = cv2.imread(self._val_info[img_id])
            if image is None:
                continue

            image_rgb = image[:, :, ::-1]
            tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float()

            with torch.no_grad():
                outputs = model([{
                    "image": tensor.to("cuda"),
                    "height": image.shape[0],
                    "width": image.shape[1],
                }])

            inst = outputs[0]["instances"]
            pred_boxes = inst.pred_boxes.tensor.cpu().numpy()
            pred_scores = inst.scores.cpu().numpy()
            pred_classes = inst.pred_classes.cpu().numpy()

            mask = pred_scores >= self._confidence_threshold
            pred_boxes = pred_boxes[mask]
            pred_classes = pred_classes[mask]

            gt_matched = [False] * len(annotations)

            for pbox, pcls in zip(pred_boxes, pred_classes):
                best_iou, best_gi = 0, -1
                for gi, ann in enumerate(annotations):
                    if gt_matched[gi]:
                        continue
                    iou = self._iou(pbox, ann['box'])
                    if iou > best_iou and iou >= self._iou_threshold:
                        best_iou = iou
                        best_gi = gi

                if best_gi >= 0:
                    gt_matched[best_gi] = True
                    gt_name = annotations[best_gi]['class']
                    DINO_IDX_TO_NAME = {
                        1: "black-bishop", 2: "black-king", 3: "black-knight",
                        4: "black-pawn", 5: "black-queen", 6: "black-rook",
                        7: "white-bishop", 8: "white-king", 9: "white-knight",
                        10: "white-pawn", 11: "white-queen", 12: "white-rook"
                    }
                    pred_name = DINO_IDX_TO_NAME.get(int(pcls), "")
                    total += 1
                    if pred_name == gt_name:
                        correct += 1

        return (correct / total * 100) if total > 0 else 0, correct, total

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0:
            return
        if not comm.is_main_process():
            return

        model = self.trainer.model
        model.eval()
        acc, correct, total = self._compute_cls_accuracy(model)
        model.train()

        if acc > self._best_acc:
            self._best_acc = acc
            self._checkpointer.save("model_best")
            print(f"\n[BestCheckpoint] New best cls accuracy: {correct}/{total} = {acc:.1f}% "
                  f"(iter {next_iter}) -> saved model_best.pth\n")
        else:
            print(f"\n[BestCheckpoint] Cls accuracy: {correct}/{total} = {acc:.1f}% "
                  f"(best: {self._best_acc:.1f}%) -> not saved\n")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, ["backbone"])
                and not match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": 2e-4,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["backbone"]) and p.requires_grad
            ],
            "lr": 2e-5,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": 2e-5,
        },
    ]
    optim = torch.optim.AdamW(param_dicts, 2e-4, weight_decay=1e-4)

    train_loader = instantiate(cfg.dataloader.train)
    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dino"))
    best_ckpt_hook = BestCheckpointHook(
        eval_period=cfg.train.eval_period,
        checkpointer=checkpointer,
        val_ann_path=os.path.join(data_root, "val/game2/_annotations_merged.coco.json"),
        val_img_dir=os.path.join(data_root, "val/game2"),
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            best_ckpt_hook,
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
