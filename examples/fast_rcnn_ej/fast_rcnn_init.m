function net = fast_rcnn_init(varargin)
%FAST_RCNN_INIT  Initialize a Fast-RCNN

% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.piecewise = 1;
opts.modelPath = fullfile('data', 'models','imagenet-resnet-50-dag.mat');
opts = vl_argparse(opts, varargin) ;
display(opts) ;

% Load an imagenet pre-trained cnn model.
netStruct = load(opts.modelPath);
netStruct.layers(1).inputs{1}='input';
netStruct.vars(1).name='input';

net = dagnn.DagNN.loadobj(netStruct) ;
clear netStruct ;

% net.vars(1).name='input';

nCls = 21;

% Skip pool5.
pPool5 = find(arrayfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
pparFc = find(arrayfun(@(a) strcmp(a.name, 'fc1000_filter'), net.params)==1);
fc6_size=[7 7 size(net.params(pparFc).value,3) 2048];
net.layers = net.layers([1:pPool5-1]);

net.rebuild();

% Add ROIPooling layer.
pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'res5c_relu'), net.layers)==1);
if isempty(pRelu5)
  error('Cannot find last relu before fc');
end

net.addLayer('roipool', ...
             dagnn.ROIPooling('method','max','transform',1/16,'subdivisions',[7,7],'flatten',0), ...
             {net.layers(pRelu5).outputs{1},'rois'}, ...
             'roi');
pRP = find(arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);

net.addLayer('fc6' , ...
             dagnn.Conv('size', fc6_size), ...
             net.layers(pRP).outputs{1}, ...
             'fc6', ...
             {'fc6_filter', 'fc6_bias'}) ;
net.params(end-1).value = 0.01 * randn(fc6_size,'single');
net.params(end).value = zeros(2048,1,'single');
         
net.addLayer('relu6' , ...
             dagnn.ReLU(), ...
             'fc6', ...
             'relu6') ;
         
% net.addLayer('drop6', ...
%              dagnn.DropOut(), ...
%              'relu6', ...
%              'drop6') ;
         
net.addLayer('fc7' , ...
             dagnn.Conv('size', [1 1 2048 4096]), ...
             'relu6', ...
             'fc7', ...
             {'fc7_filter', 'fc7_bias'}) ;
net.params(end-1).value = 0.01 * randn(1,1,2048,4096,'single');
net.params(end).value = zeros(4096,1,'single');
         
net.addLayer('relu7' , ...
             dagnn.ReLU(), ...
             'fc7', ...
             'relu7') ;

% net.addLayer('drop7', ...
%             dagnn.DropOut(), ...
%             'relu7', ...
%             'drop7') ;

% Add class predict layer.
pFc7 = find(arrayfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);
pparFc7 = find(arrayfun(@(a) strcmp(a.name, 'fc7_filter'), net.params)==1);
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 size(net.params(pparFc7).value,4) nCls], 'hasBias', true), ...
             net.layers(pFc7).outputs{1}, ...
             'prediction', ...
             {'prediction_filter', 'prediction_bias'}) ;

net.params(end-1).value = 0.01 * randn(1,1,size(net.params(pparFc7).value,4),nCls,'single');
net.params(end).value = zeros(1,nCls,'single');

% Add softmax loss layer.
pFc8 = find(arrayfun(@(a) strcmp(a.name, 'prediction'), net.layers)==1);
net.addLayer('losscls',dagnn.Loss(), ...
             {net.layers(pFc8).outputs{1},'label'}, ...
             'losscls',{});

% Add bbox regression layer.
if opts.piecewise
  pparCls = find(arrayfun(@(a) strcmp(a.name, 'prediction_filter'), net.params)==1);
  net.addLayer('predbbox', ...
               dagnn.Conv('size',[1 1 size(net.params(pparCls).value,3) 84],'hasBias', true), ...
               net.layers(pFc7).outputs{1}, ...
               'predbbox',{'predbbox_filter','predbbox_bias'});
  
  display(net.params(pparCls));
  net.params(end-1).value = 0.001 * randn(1,1,size(net.params(pparCls).value,3),84,'single');
  net.params(end).value = zeros(1,84,'single');
  
  display(net.params(end-1));
  net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
               {'predbbox','targets','instance_weights'}, ...
               'lossbbox',{});
end

net.rebuild();

% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = 2;
end

% Change image-mean as in fast-rcnn code
net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';

net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
    'sofa', 'train', 'tvmonitor', 'background' };
  
net.meta.classes.description = {};
