use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::Tensor,
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

/// The IoU (Intersection over Union) accuracy metric for object detection.
#[derive(Default)]
pub struct IoUMetric<B: Backend> {
    state: NumericMetricState,
    // iou_threshold: f64, // Minimum IoU to consider a detection as correct.
    _b: PhantomData<B>,
}

/// The [IoU accuracy metric](IoUMetric) input type.
pub struct IoUInput<B: Backend> {
    pub output: Tensor<B, 3>,  // [batch_size, num_boxes, 4] predicted boxes
    pub targets: Tensor<B, 3>, // [batch_size, num_boxes, 4] ground-truth boxes
}

/// Implements the IoU calculation logic.
impl<B: Backend> IoUMetric<B> {
    /// Creates the IoU metric with the specified threshold.
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            // iou_threshold,
            _b: PhantomData,
        }
    }

    // [x, y, w, h] represents [top-left x, top-left y, width, height]
    fn calculate_iou(&self, pred_box: &[f64], target_box: &[f64]) -> f64 {
        // Predicted box coordinates
        let (pred_x, pred_y, pred_w, pred_h) = (pred_box[0], pred_box[1], pred_box[2], pred_box[3]);

        // Target box coordinates
        let (target_x, target_y, target_w, target_h) =
            (target_box[0], target_box[1], target_box[2], target_box[3]);

        // Coordinates of the intersection rectangle
        let inter_x1 = pred_x.max(target_x);
        let inter_y1 = pred_y.max(target_y);
        let inter_x2 = (pred_x + pred_w).min(target_x + target_w);
        let inter_y2 = (pred_y + pred_h).min(target_y + target_h);

        // Intersection width and height
        let inter_w = (inter_x2 - inter_x1).max(0.0); // Ensure non-negative
        let inter_h = (inter_y2 - inter_y1).max(0.0); // Ensure non-negative

        // Area of intersection
        let inter_area = inter_w * inter_h;

        // Areas of the predicted and target boxes
        let pred_area = pred_w * pred_h;
        let target_area = target_w * target_h;

        // Area of union: A∪B = A + B - A∩B
        let union_area = pred_area + target_area - inter_area;

        // IoU = Area of Intersection / Area of Union
        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    /// Update the metric with a batch of predictions and targets.
    fn update_batch(&mut self, pred_boxes: Tensor<B, 3>, target_boxes: Tensor<B, 3>) -> f64 {
        let [batch_size, num_boxes, _] = pred_boxes.dims();
        let mut correct_detections = 0.0;
        let pred_boxes = pred_boxes.flatten::<1>(0, 2).to_data();
        let pred_boxes = pred_boxes.as_slice::<f64>().unwrap();

        let target_boxes = target_boxes.flatten::<1>(0, 2).to_data();
        let target_boxes = target_boxes.as_slice::<f64>().unwrap();

        for (pred_box, target_box) in pred_boxes.chunks(4).zip(target_boxes.chunks(4)) {
            correct_detections += self.calculate_iou(pred_box, target_box);
        }

        // for i in 0..batch_size {
        //     let pred_batch = pred_boxes.index(i).reshape([num_boxes, 4]);
        //     let target_batch = target_boxes.index(i).reshape([num_boxes, 4]);
        //
        //     let iou = self.calculate_iou(pred_batch, target_batch); // [num_boxes]
        //     let correct = iou.ge_elem(self.iou_threshold).int().sum().elem::<f64>();
        //
        //     correct_detections += correct;
        // }
        //
        correct_detections / (batch_size as f64 * num_boxes as f64)
    }
}

impl<B: Backend> Metric for IoUMetric<B> {
    const NAME: &'static str = "IoU Accuracy";

    type Input = IoUInput<B>;

    fn update(&mut self, input: &IoUInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let accuracy = self.update_batch(input.output.clone(), input.targets.clone());

        self.state.update(
            100.0 * accuracy,
            input.output.dims()[0], // Batch size
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for IoUMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
