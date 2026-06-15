import pandas as pd
import matplotlib.pyplot as plt


DINO_CSV = "dinov2_threshold_sweep.csv"
YOLO_CSV = "yolo_confidence_sweep.csv"

dino = pd.read_csv(DINO_CSV)
yolo = pd.read_csv(YOLO_CSV)


# =========================
# IoU
# =========================

# plt.figure(figsize=(8, 5))
#
# plt.plot(
#     dino["threshold"],
#     dino["dataset_iou"],
#     marker="o",
#     label="YD5 (DINOv2 + NN)"
# )
#
# plt.plot(
#     yolo["conf"],
#     yolo["dataset_iou"],
#     marker="s",
#     label="YS5 (YOLO-Seg)"
# )
#
# plt.xlabel("Threshold / Confidence")
# plt.ylabel("Dataset IoU")
# plt.title("IoU Comparison")
# plt.legend()

# plt.savefig("comparison_iou.png", dpi=300, bbox_inches="tight")
# plt.close()
#
#
# # =========================
# # Dice
# # =========================
#
# plt.figure(figsize=(8, 5))
#
# plt.plot(
#     dino["threshold"],
#     dino["dataset_dice"],
#     marker="o",
#     label="DINOv2"
# )
#
# plt.plot(
#     yolo["conf"],
#     yolo["dataset_dice"],
#     marker="s",
#     label="YOLO-Seg"
# )
#
# plt.xlabel("Threshold / Confidence")
# plt.ylabel("Dataset Dice")
# plt.title("Dice Comparison")
# plt.legend()
#
# plt.savefig("comparison_dice.png", dpi=300, bbox_inches="tight")
# plt.close()
#
#
# # =========================
# # Precision
# # =========================
#
# plt.figure(figsize=(8, 5))
#
# plt.plot(
#     dino["threshold"],
#     dino["dataset_precision"],
#     marker="o",
#     label="DINOv2"
# )
#
# plt.plot(
#     yolo["conf"],
#     yolo["dataset_precision"],
#     marker="s",
#     label="YOLO-Seg"
# )
#
# plt.xlabel("Threshold / Confidence")
# plt.ylabel("Precision")
# plt.title("Precision Comparison")
# plt.legend()
#
# plt.savefig("comparison_precision.png", dpi=300, bbox_inches="tight")
# plt.close()
#
#
# # =========================
# # Recall
# # =========================
#
# plt.figure(figsize=(8, 5))
#
# plt.plot(
#     dino["threshold"],
#     dino["dataset_recall"],
#     marker="o",
#     label="DINOv2"
# )
#
# plt.plot(
#     yolo["conf"],
#     yolo["dataset_recall"],
#     marker="s",
#     label="YOLO-Seg"
# )
#
# plt.xlabel("Threshold / Confidence")
# plt.ylabel("Recall")
# plt.title("Recall Comparison")
# plt.legend()
#
# plt.savefig("comparison_recall.png", dpi=300, bbox_inches="tight")
# plt.close()


# =========================
# Precision-Recall-Kurve
# =========================

plt.figure(figsize=(6, 6))

plt.plot(
    dino["dataset_recall"],
    dino["dataset_precision"],
    marker="o",
    label="D (DINOv2 + NN)"
)

plt.plot(
    yolo["dataset_recall"],
    yolo["dataset_precision"],
    marker="s",
    label="Y (YOLO-Seg)"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
#plt.title("Precision-Recall Comparison")
plt.legend()

plt.savefig("comparison_precision_recall_baseline.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# BESTE WERTE
# =========================

best_dino_iou = dino.loc[dino["dataset_iou"].idxmax()]
best_yolo_iou = yolo.loc[yolo["dataset_iou"].idxmax()]

summary = pd.DataFrame([
    {
        "Model": "DINOv2",
        "Parameter": best_dino_iou["threshold"],
        "IoU": best_dino_iou["dataset_iou"],
        "Dice": best_dino_iou["dataset_dice"],
        "Precision": best_dino_iou["dataset_precision"],
        "Recall": best_dino_iou["dataset_recall"]
    },
    {
        "Model": "YOLO-Seg",
        "Parameter": best_yolo_iou["conf"],
        "IoU": best_yolo_iou["dataset_iou"],
        "Dice": best_yolo_iou["dataset_dice"],
        "Precision": best_yolo_iou["dataset_precision"],
        "Recall": best_yolo_iou["dataset_recall"]
    }
])

summary.to_csv("best_model_comparison.csv", index=False)

print(summary)

print("\nGespeicherte Plots:")
print("comparison_iou.png")
print("comparison_dice.png")
print("comparison_precision.png")
print("comparison_recall.png")
print("comparison_precision_recall.png")