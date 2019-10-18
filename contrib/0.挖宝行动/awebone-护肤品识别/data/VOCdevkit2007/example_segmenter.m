% example_segmenter Segmentation algorithm based on detection results.
%
% This segmenter requires that some detection results are present in
% 'Results' e.g. by running 'example_detector'.
%
% Segmentations are generated from detection bounding boxes.
function example_segmenter

VOCinit
create_segmentations_from_detections('comp3',1)
VOCevalseg(VOCopts,'comp3');
