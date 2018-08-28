% --------------------------------------------------------------------
function net = deployFRCNN
% --------------------------------------------------------------------
expDir    = fullfile(vl_rootnn, 'data', 'fast-rcnn-resnet50a-0.05-pascal07') ;
imdbPath  = fullfile(expDir, 'imdb.mat');

modelPath = @(ep) fullfile(expDir, sprintf('net-epoch-%d.mat', ep));

list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
start = max([epoch 0]) ;
start = 9 ;

fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
load(modelPath(start), 'net', 'state', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

fprintf('Loading imdb...\n');
imdb = load(imdbPath) ;

% function net = deployFRCNN(net)
for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end

net.rebuild();

pfc8 = net.getLayerIndex('prediction') ;
net.addLayer('probcls',dagnn.SoftMax(),net.layers(pfc8).outputs{1},...
  'probcls',{});

net.vars(net.getVarIndex('probcls')).precious = true ;

idxBox = net.getLayerIndex('predbbox') ;
if ~isnan(idxBox)
  net.vars(net.layers(idxBox).outputIndexes(1)).precious = true ;
  % incorporate mean and std to bbox regression parameters
  blayer = net.layers(idxBox) ;
  filters = net.params(net.getParamIndex(blayer.params{1})).value ;
  biases = net.params(net.getParamIndex(blayer.params{2})).value ;
  
  boxMeans = single(imdb.boxes.bboxMeanStd{1}');
  boxStds = single(imdb.boxes.bboxMeanStd{2}');
  
  net.params(net.getParamIndex(blayer.params{1})).value = ...
    bsxfun(@times,filters,...
    reshape([boxStds(:)' zeros(1,4,'single')]',...
    [1 1 1 4*numel(net.meta.classes.name)]));

  biases = biases .* [boxStds(:)' zeros(1,4,'single')];
  
  net.params(net.getParamIndex(blayer.params{2})).value = ...
    bsxfun(@plus,biases, [boxMeans(:)' zeros(1,4,'single')]);
end

net.mode = 'test' ;

ModelName= sprintf('net-deployed-epoch%d.mat',start) ; 
saveModelPath = fullfile(expDir, ModelName);
net_ = net.saveobj() ;
save(saveModelPath, '-struct', 'net_') ;
clear net_ ;