function net = fast_rcnn_init_no_conv5(varargin)
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
is101 = 0;

net = dagnn.DagNN.loadobj(netStruct) ;
clear netStruct ;

% net.vars(1).name='input';

nCls = 21;

% Skip pool5. Add ROIPooling layer.
pConv4 = find(arrayfun(@(a) strcmp(a.name, 'res4f_relu'), net.layers)==1);
pparConv4 = find(arrayfun(@(a) strcmp(a.name, 'res4f_branch2c_filter'), net.params)==1);
if isempty(pConv4)
  is101 = 1;
  pConv4 = find(arrayfun(@(a) strcmp(a.name, 'res4b22_relu'), net.layers)==1);
  pparConv4 = find(arrayfun(@(a) strcmp(a.name, 'res4b22_branch2c_filter'), net.params)==1);
end
if isempty(pConv4)
  error('Cannot find last relu before fc');
end

% fc6_size=[7 7 size(net.params(pparConv4).value,4) 2048];
% net.layers = net.layers(1:pConv4);

net.rebuild();

net.addLayer('roipool', ...
             dagnn.ROIPooling('method','max','transform',1/16,'subdivisions',[7,7],'flatten',0), ...
             {net.layers(pConv4).outputs{1},'rois'}, ...
             'roi');
         
%modify res5a stride
pRP = find(arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);
pBn5 = find(arrayfun(@(a) strcmp(a.name, 'bn5a_branch1'), net.layers)==1);
pparConv5 = find(arrayfun(@(a) strcmp(a.name, 'res5a_branch1_filter'), net.params)==1);
pparConv5_2 = find(arrayfun(@(a) strcmp(a.name, 'res5a_branch2a_filter'), net.params)==1);
Conv5_filter = net.params(pparConv5).value;
Conv5_2_filter=net.params(pparConv5_2).value;
net.layers = [net.layers(1:pConv4) net.layers(pRP) net.layers(pBn5) net.layers(pBn5+2:end-1)];
net.rebuild();

pRP = find(arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);
pBn5 = find(arrayfun(@(a) strcmp(a.name, 'bn5a_branch1'), net.layers)==1);
net.addLayer('res5a_branch1',...
             dagnn.Conv('size', [1 1 size(net.params(pparConv5).value,4) 2048], ...
                        'stride', 1, ...
                        'pad', 0, ...
                        'hasBias', false), ...
             net.layers(pRP).outputs{1}, ...
             'res5a_branch1', ...
             {'res5a_branch1_filter'});
net.params(end).value=Conv5_filter;
net.addLayer('res5a_branch2a',...
             dagnn.Conv('size', [1 1 size(net.params(pparConv5).value,4) 512], ...
                        'stride', 1, ...
                        'pad', 0, ...
                        'hasBias', false), ...
             net.layers(pRP).outputs{1}, ...
             'res5a_branch2a', ...
             {'res5a_branch2a_filter'});
net.params(end).value=Conv5_2_filter; 
net.layers = [net.layers(1:end-4) net.layers(end-1) net.layers(end)];
% pparConv5 = find(arrayfun(@(a) strcmp(a.name, 'bn5a_branch1_mult'), net.params)==1);
% pparConv5_2 = find(arrayfun(@(a) strcmp(a.name, 'bn5a_branch2a_mult'), net.params)==1);
% net.params = [net.params(1:pparConv5-1) net.params(end-1) net.params(pparConv5:pparConv5_2-1) net.params(end) net.params(pparConv5_2:end-2)];
net.rebuild();

% remove Batch normalization
pBn = find(arrayfun(@(a) strncmp(a.name, 'bn5', 3), net.layers)==1);
for i=1:numel(pBn)
  if strcmp(net.layers(pBn(i)+1).inputs{1}, net.layers(pBn(i)).outputs{1})==1
    net.layers(pBn(i)+1).inputs{1} = net.layers(pBn(i)).inputs{1};
  end
end

pSum1 = find(arrayfun(@(a) strcmp(a.name, 'res5a'), net.layers)==1);
pSum2 = find(arrayfun(@(a) strcmp(a.name, 'res5b'), net.layers)==1);
pSum3 = find(arrayfun(@(a) strcmp(a.name, 'res5c'), net.layers)==1);
net.layers(pSum1).inputs{1}=net.layers(end-1).outputs{1};
net.layers(pSum1).inputs{2}=net.layers(pBn(4)-1).outputs{1};
net.layers(pSum2).inputs{2}=net.layers(pBn(7)-1).outputs{1};
net.layers(pSum3).inputs{2}=net.layers(pBn(10)-1).outputs{1};

netIdx=[];
j=1;
for i=1:numel(net.layers)
  if j<=numel(pBn)
    if i==pBn(j)
      j=j+1;
      continue
    end
  end
  netIdx = [netIdx i];
end
net.layers = net.layers(netIdx);
net.rebuild();

% add fc layers
% net.addLayer('fc6' , ...
%              dagnn.Conv('size', [7 7 2048 2048]), ...
%              net.layers(end).outputs{1}, ...
%              'fc6', ...
%              {'fc6_filter', 'fc6_bias'}) ;
% net.params(end-1).value = 0.01 * randn(7,7,2048,2048,'single');
% net.params(end).value = zeros(2048,1,'single');
%          
% net.addLayer('relu6' , ...
%              dagnn.ReLU(), ...
%              'fc6', ...
%              'relu6') ;
%          
% net.addLayer('fc7' , ...
%              dagnn.Conv('size', [1 1 2048 4096]), ...
%              'relu6', ...
%              'fc7', ...
%              {'fc7_filter', 'fc7_bias'}) ;
% net.params(end-1).value = 0.01 * randn(1,1,2048,4096,'single');
% net.params(end).value = zeros(4096,1,'single');
%          
% net.addLayer('relu7' , ...
%              dagnn.ReLU(), ...
%              'fc7', ...
%              'relu7') ;

% Add class predict layer.
% pFc7 = find(arrayfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);
% pparFc7 = find(arrayfun(@(a) strcmp(a.name, 'fc7_filter'), net.params)==1);

% modify learning rate
pparBn = find(arrayfun(@(a) strcmp(a.name, 'bn5a_branch1_mult'), net.params)==1) ;
if isempty(pparBn)
    pparBn = find(arrayfun(@(a) strcmp(a.name, 'res5a_branch2b_filter'), net.params)==1);
end
for i= pparBn:numel(net.params)
  if numel(strfind(net.params(i).name, 'moments'))==0
    net.params(i).learningRate = 2.5 ;
  end
end

pFc7 = find(arrayfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
pparFc7 = find(arrayfun(@(a) strcmp(a.name, 'res5c_branch2c_filter'), net.params)==1);
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 size(net.params(pparFc7).value,4) nCls], 'hasBias', true), ...
             net.layers(pFc7).outputs{1}, ...
             'prediction', ...
             {'prediction_filter', 'prediction_bias'}) ;
net.params(end-1).value = 0.01 * randn(1,1,size(net.params(pparFc7).value,4),nCls,'single');
% net.params(end-1).learningRate = 2;
net.params(end).value = zeros(1,nCls,'single');
net.params(end).learningRate = 2;
net.params(end).weightDecay = 0;

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
  % net.params(end-1).learningRate = 5;
  net.params(end).value = zeros(1,84,'single');
  net.params(end).learningRate = 2;
  net.params(end).weightDecay = 0;

  
  display(net.params(end-1));
  net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
               {'predbbox','targets','instance_weights'}, ...
               'lossbbox',{});
end

net.rebuild();

% No decay for bias and set learning rate to 2
% for i=2:2:numel(net.params)
%   net.params(i).weightDecay = 0;
%   net.params(i).learningRate = 2;
% end

% Change image-mean as in fast-rcnn code
net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';

net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
    'sofa', 'train', 'tvmonitor', 'background' };
  
net.meta.classes.description = {};
end
