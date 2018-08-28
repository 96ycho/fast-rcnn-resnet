function imdb = cnn_setup_data_voc12

dataDir = fullfile(vl_rootnn, 'data') ;
% Initialize VOC options
VOCinit ;
VOCopts.dataset = 'VOC2012';
VOCopts.imgsetpath = fullfile(dataDir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', '%s.txt');
VOCopts.annopath  = fullfile(dataDir, 'VOCdevkit', 'VOC2012' ,'Annotations', '%s.xml');

imdb.classes.name = VOCopts.classes ;
imdb.classes.description = VOCopts.classes ;
imdb.imageDir = fullfile(opts.dataDir, 'VOCdevkit', 'VOC2012' ,'JPEGImages') ;

% -------------------------------------------------------------------------
%                                                                    Images
% -------------------------------------------------------------------------

k = 0 ;
for thisSet = {'train', 'val'}
  thisSet = char(thisSet) ;

  fprintf('Loading PASCAL VOC %s set\n', thisSet) ;
  VOCopts.testset = thisSet ;

  [gtids,t]=textread(sprintf(VOCopts.imgsetpath,thisSet),'%s %d');
    
  k = k + 1 ;
  imdb_.images.name{k} = strcat(gtids,'.jpg');
  imdb_.images.set{k}  = k * ones(size(imdb_.images.name{k}));
  imdb_.images.size{k} = zeros(numel(imdb_.images.name{k}),2);
  imdb_.boxes.gtbox{k} = cell(size(imdb_.images.name{k}));
  imdb_.boxes.gtlabel{k} = cell(size(imdb_.images.name{k}));

  % Load ground truth objects
  for i=1:length(gtids)
    % Read annotation.
    rec=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    %rec=PASreadrecord(gtids{i}{1});
    
    imdb_.images.size{k}(i,:) = rec.imgsize(1:2);

    % extract objects of class
    BB = vertcat(rec.objects(:).bbox);
    diff = vertcat(rec.objects(:).difficult);
    if opts.useDifficult
      diff = 0 * diff ;
    end
    [~,label]=ismember({rec.objects(:).class},VOCopts.classes);

    if ~isempty(BB)
      BB = BB(~diff,:);
      imdb_.boxes.gtbox{k}{i} = BB;
      assert(all(BB(:,3)<=rec.imgsize(1)));
      assert(all(BB(:,4)<=rec.imgsize(2)));
      imdb_.boxes.gtlabel{k}{i} = label(~diff)';
    end
  end
end

imdb.images.name = vertcat(imdb_.images.name{:}) ;
imdb.images.size = vertcat(imdb_.images.size{:}) ;
imdb.images.set  = vertcat(imdb_.images.set{:}) ;
imdb.boxes.gtbox = vertcat(imdb_.boxes.gtbox{:}) ;
imdb.boxes.gtlabel = vertcat(imdb_.boxes.gtlabel{:}) ;

% -------------------------------------------------------------------------
%                                                                   Flipped
% -------------------------------------------------------------------------

trainval = (imdb.images.set==1 | imdb.images.set==2);
imdb.boxes.flip = zeros(size(imdb.images.name));

% Add flipped
if opts.addFlipped
  trainval = (imdb.images.set <= 2) ;
  imdb.images.name = vertcat(imdb.images.name, imdb.images.name(trainval)) ;
  imdb.images.set  = vertcat(imdb.images.set, imdb.images.set(trainval)) ;
  imdb.images.size  = vertcat(imdb.images.size, imdb.images.size(trainval,:)) ;

  imdb.boxes.gtbox = vertcat(imdb.boxes.gtbox , imdb.boxes.gtbox(trainval)) ;
  imdb.boxes.gtlabel = vertcat(imdb.boxes.gtlabel, imdb.boxes.gtlabel(trainval)) ;
  imdb.boxes.flip = vertcat(imdb.boxes.flip, ones(sum(trainval),1)) ;

  for i=1:numel(imdb.boxes.gtbox)
    if imdb.boxes.flip(i)
      imf = imfinfo([imdb.imageDir filesep imdb.images.name{i}]);
      gtbox = imdb.boxes.gtbox{i} ;

      assert(all(gtbox(:,1)<=imf.Width));
      assert(all(gtbox(:,3)<=imf.Width));

      gtbox(:,1) = imf.Width - gtbox(:,3) + 1;
      gtbox(:,3) = imf.Width - imdb.boxes.gtbox{i}(:,1) + 1;
      imdb.boxes.gtbox{i} = gtbox;
    end
  end
end

% -------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------

[~,si] = sort(imdb.images.name) ;
imdb.images.name = imdb.images.name(si) ;
imdb.images.set = imdb.images.set(si) ;
imdb.images.size = imdb.images.size(si,:) ;
imdb.boxes.gtbox = imdb.boxes.gtbox(si)' ;
imdb.boxes.gtlabel = imdb.boxes.gtlabel(si) ;
imdb.boxes.flip = imdb.boxes.flip(si) ;
