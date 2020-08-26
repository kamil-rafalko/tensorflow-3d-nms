def nms(boxes, scores):
    print(boxes)
    print(scores)
    indices = tf.image.non_max_suppression(
        boxes, scores, self.proposal_count,
        self.nms_threshold, name="rpn_non_max_suppression")

    all_boxes = tf.identity(boxes)
    all_scores = tf.identity(scores)
    sorted_args = tf.argsort(all_scores)
    final_boxes = []

    def nms_step(boxes, scores, sorted_args, result):
        candidate = sorted_args[0, :]

        def check_box(box, boxes, scores):
            zA = tf.math.maximum(candidate[0], box[0])
            yA = tf.math.maximum(candidate[1], box[1])
            xA = tf.math.maximum(candidate[2], box[2])
            zB = tf.math.maximum(candidate[3], box[3])
            yB = tf.math.maximum(candidate[4], box[4])
            xB = tf.math.maximum(candidate[5], box[5])

            interArea = tf.math.maximum(0, xB - xA) * tf.math.maximum(0, yB - yA) * tf.math.maximum(0, zB - zA)

            boxAArea = (candidate[3] - candidate[0]) * (candidate[4] - candidate[1]) * (candidate[5] - candidate[2])
            boxBArea = (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])

            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > self.nms_threshold:
                boxes = boxes[1:, :]

        tf.map_fn(lambda box: check_box(box, boxes, scores), boxes)

    tf.while_loop(
        lambda b, s, sa, f: tf.greater(b.shape[0], 0),
        lambda b, s, sa, f: nms_step(b, s, sa, f),
        (all_boxes, all_scores, sorted_args, final_boxes)
    )

    while (len(all_boxes) != 0):
        highest_score_idx = tf.argmax(all_scores)
        final_boxes

    proposals = tf.gather(boxes, indices)
    # Pad if needed
    padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
    proposals = tf.pad(proposals, [(0, padding), (0, 0)])
    return proposals