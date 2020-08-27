import tensorflow as tf

def nms(boxes, scores, proposal_count, nms_threshold):
    print(boxes)
    print(scores)

    indices = tf.image.non_max_suppression(
        boxes, scores, proposal_count,
        nms_threshold, name="rpn_non_max_suppression")

    all_boxes = tf.identity(boxes)
    all_scores = tf.identity(scores)
    sorted_args = tf.argsort(all_scores)
    final_boxes = []

    def nms_step(boxes, scores, sorted_args, result):
        candidate = sorted_args[0, :]

        def check_box(box, boxes, scores):
            def iou(a, b):
                zA = tf.math.maximum(a[0], b[0])
                yA = tf.math.maximum(a[1], b[1])
                xA = tf.math.maximum(a[2], b[2])
                zB = tf.math.maximum(a[3], b[3])
                yB = tf.math.maximum(a[4], b[4])
                xB = tf.math.maximum(a[5], b[5])

                interArea = tf.math.maximum(0, xB - xA) * tf.math.maximum(0, yB - yA) * tf.math.maximum(0, zB - zA)

                boxAArea = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
                boxBArea = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])

                return interArea / (boxAArea + boxBArea - interArea)

            iou = iou(candidate, box)
            if iou > nms_threshold:
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