import torch
import torch.nn.functional as F

from .dn_criterion import DINOCriterion


def sigmoid_focal_loss_weighted(inputs, targets, class_weights, num_boxes, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if class_weights is not None:
        weights = class_weights.view(1, 1, -1).to(loss.device)
        weight_mask = targets * weights + (1 - targets) * 1.0
        loss = loss * weight_mask

    return loss.mean(1).sum() / num_boxes


class WeightedDINOCriterion(DINOCriterion):

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
        class_weights=None,
    ):
        super().__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
            two_stage_binary_cls=two_stage_binary_cls,
        )

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        if self.loss_class_type == "focal_loss":
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            loss_class = (
                sigmoid_focal_loss_weighted(
                    src_logits,
                    target_classes_onehot,
                    class_weights=self.class_weights,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        else:
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )

        losses = {"loss_class": loss_class}
        return losses
