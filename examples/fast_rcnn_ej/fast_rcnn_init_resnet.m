function net=fast_rcnn_init_resnet(varargin)

opts.piecewise=1;
opts.layer=50;
opts.roisize=7;
% opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
lastAdded.depth = 3 ;

function Conv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.relu = true ;
  args.downsample = false ;
  args.bias = false ;
  args = vl_argparse(args, varargin) ;
  if args.downsample, stride = 2 ; else stride = 1 ; end
  if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end
  net.addLayer([name  '_conv'], ...
               dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
                          'stride', stride, ....
                          'pad', (ksize - 1) / 2, ...
                          'hasBias', args.bias), ...
               lastAdded.var, ...
               [name '_conv'], ...
               pars) ;
  net.addLayer([name '_bn'], ...
               dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
               [name '_conv'], ...
               [name '_bn'], ...
               {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
  lastAdded.depth = depth ;
  lastAdded.var = [name '_bn'] ;
  if args.relu
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 lastAdded.var, ...
                 [name '_relu']) ;
    lastAdded.var = [name '_relu'] ;
  end
end

% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1', 7, 64, ...
     'relu', true, ...
     'bias', false, ...
     'downsample', true) ;
 
net.addLayer(...
  'conv1_pool' , ...
  dagnn.Pooling('poolSize', [3 3], ...
                'stride', 2, ...
                'pad', 1,  ...
                'method', 'max'), ...
  lastAdded.var, ...
  'conv1') ;
lastAdded.var = 'conv1' ;

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------
function inter_layer(layerNum)
  for s = 2:layerNum
    switch s
      case 2, sectionLen = 3 ;
      case 3 
          if strcmp(opts.layer,'50')
            sectionLen = 4 ; 
          else
            sectionLen = 8 ;
          end
      case 4 
          if strcmp(opts.layer,'50')
            sectionLen = 6 ; 
          else
            sectionLen = 23 ;
          end
      case 5, sectionLen = 3 ;
    end

    % -----------------------------------------------------------------------
    % Add intermediate segments for each section
    for l = 1:sectionLen
      depth = 2^(s+4) ;
      sectionInput = lastAdded ;
      name = sprintf('conv%d_%d', s, l)  ;

      % Optional adapter layer
      if l == 1
        Conv([name '_adapt_conv'], 1, 2^(s+6), 'downsample', s >= 3, 'relu', false) ;
      end
      sumInput = lastAdded ;

      % ABC: 1x1, 3x3, 1x1; downsample if first segment in section from
      % section 2 onwards.
      lastAdded = sectionInput ;
      Conv([name 'a'], 1, 2^(s+4)) ;
      Conv([name 'b'], 3, 2^(s+4), 'downsample', (s >= 3) & l == 1) ;
      Conv([name 'c'], 1, 2^(s+6), 'relu', false) ;

      % Sum layer
      net.addLayer([name '_sum'] , ...
                   dagnn.Sum(), ...
                   {sumInput.var, lastAdded.var}, ...
                   [name '_sum']) ;
      net.addLayer([name '_relu'] , ...
                   dagnn.ReLU(), ...
                   [name '_sum'], ...
                   name) ;
      lastAdded.var = name ;
    end
  end
end

inter_layer(5);
fc6_size = [opts.roisize opts.roisize 2048 2048] ;

net.addLayer('fc6' , ...
             dagnn.Conv('size', fc6_size), ...
             lastAdded.var, ...
             'fc6', ...
             {'fc6_f', 'fc6_b'}) ;
         
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
             {'fc7_f', 'fc7_b'}) ;
         
net.addLayer('relu7' , ...
             dagnn.ReLU(), ...
             'fc7', ...
             'relu7') ;

% net.addLayer('drop7', ...
%             dagnn.DropOut(), ...
%             'relu7', ...
%             'drop7') ;

nCls = 21;         

%Init parameters randomly
net.initParams();

% Add ROIPooling layer.
vggdeep = false;
pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5'), net.layers)==1);
if isempty(pRelu5)
  vggdeep = true;
  % if strcmp(opts.layer, '50B')
  %   lastname='conv4_6_relu' ;
  lastname='conv5_3_relu' ;
  % end
  pRelu5 = find(arrayfun(@(a) strcmp(a.name, lastname), net.layers)==1);
  if isempty(pRelu5)
    error('Cannot find last relu before fc');
  end
end

net.addLayer('roipool', ...
             dagnn.ROIPooling('method','max','transform',1/16,'subdivisions',[opts.roisize,opts.roisize],'flatten',0), ...
             {net.layers(pRelu5).outputs{1},'rois'},...
             'xRP');

pFc6 = find(arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);
pRP = find(arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);
net.layers(pFc6).inputs{1} = net.layers(pRP).outputs{1};

% Add class predict layer.
pFc7 = find(arrayfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);
pparFc7 = find(arrayfun(@(a) strcmp(a.name, 'fc7_f'), net.params)==1);
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 size(net.params(pparFc7).value,4) nCls], 'hasBias', true), ...
             net.layers(pFc7).outputs{1}, ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;

net.params(end-1).value = 0.01 * randn(1,1,size(net.params(pparFc7).value,4),nCls,'single');
net.params(end).value = zeros(1,nCls,'single');

% Add softmax loss layer.
pFc8 = find(arrayfun(@(a) strcmp(a.name, 'prediction'), net.layers)==1);
net.addLayer('losscls',...
             dagnn.Loss(), ...
             {'prediction','label'}, ...
             'losscls',{});

% Add bbox regression layer.
if opts.piecewise
  pparFc8 = find(arrayfun(@(a) strcmp(a.name, 'prediction_f'), net.params)==1);
  net.addLayer('predbbox',dagnn.Conv('size',[1 1 size(net.params(pparFc8).value,3) 84],'hasBias', true), ...
               net.layers(pFc7).outputs{1},...
               'predbbox',...
               {'predbbox_f','predbbox_b'});
   
  display(net.params(pparFc8));
  net.params(end-1).value = 0.001 * randn(1,1,size(net.params(pparFc8).value,3),84,'single');
  net.params(end).value = zeros(1,84,'single');

  net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
               {'predbbox','targets','instance_weights'}, ...
               'lossbbox',{});
end

net.rebuild();

net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';

net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', ...
    'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ...
    'sofa', 'train', 'tvmonitor', 'background' };
  
net.meta.classes.description = {};
         
end