function pred_boxes = bbox_transform_inv(boxes, deltas)
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% -------------------------------------------------------------------------
% resize bbox
% -------------------------------------------------------------------------

function boxOut = bbox_resize(boxIn, scale1, scale2, szOut)

if isempty(boxIn), boxOut = []; return; end

boxOut(:,1) = scale1 * (boxIn(:,1)-1) + 1;
boxOut(:,2) = scale2 * (boxIn(:,2)-1) + 1;
boxOut(:,3) = scale1 * (boxIn(:,3)-1) + 1;
boxOut(:,4) = scale2 * (boxIn(:,4)-1) + 1;

boxOut = [max(1,round(boxOut(:,1))),...
  max(1,round(boxOut(:,2))),...
  min(szOut(2),round(boxOut(:,3))),...
  min(szOut(1),round(boxOut(:,4)))];

end

if isempty(boxes), return; end 

% bbox resize
% factor1 = imgSize/originSize(1);
% factor2 = imgSize/originSize(2);
% boxes = bbox_resize(boxes,factor2,factor1,[imgSize imgSize]);

nw = boxes(:,3) - boxes(:,1);
nh = boxes(:,4) - boxes(:,2);
nctrx = boxes(:,1) + 0.5 * nw;
nctry = boxes(:,2) + 0.5 * nh;

dx = deltas(:,1);
dy = deltas(:,2);
dw = deltas(:,3);
dh = deltas(:,4);

pred_ctr_x = dx .* nw + nctrx;
pred_ctr_y = dy .* nh + nctry;
pred_w = exp(dw) .* nw;
pred_h = exp(dh) .* nh;

pred_boxes = zeros(size(deltas), 'like', deltas);
% x1
pred_boxes(:, 1) = pred_ctr_x - 0.5 * pred_w;
% y1
pred_boxes(:, 2) = pred_ctr_y - 0.5 * pred_h;
% x2
pred_boxes(:, 3) = pred_ctr_x + 0.5 * pred_w;
% y2
pred_boxes(:, 4) = pred_ctr_y + 0.5 * pred_h;

% return to original size
% factor1 = originSize(1)/imgSize;
% factor2 = originSize(2)/imgSize;
% pred_boxes = bbox_resize(pred_boxes,factor2,factor1,originSize);

end
