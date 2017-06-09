% Computing precision, recall and f-measure for bungalows dataset
caffe.reset_all();
clear;clc;
addpath(genpath('../../caffe'));

caffe.set_mode_gpu();
caffe.set_device(0);

% model
cnnmodel = './deploy.prototxt';
cnnweights = './train_iter_80000.caffemodel';

acceptedSize = [480 360];

mean_data(:,:,3) = repmat(96.14,acceptedSize); %B
mean_data(:,:,2) = repmat(112.16,acceptedSize); %G
mean_data(:,:,1) = repmat(99.34,acceptedSize); %R
mean_data = single(mean_data);

net = caffe.Net(cnnmodel, cnnweights, 'test'); % create net and load weights

imagePath = '/home/sepideh/Documents/SegNet/bungalows/test';
labelPath = '/home/sepideh/Documents/SegNet/bungalows/testannot';
imageType = '*.jpg';
images = dir([imagePath '/' imageType]);
labels = dir([labelPath '/' '*.png']);
nImages = length(images);

precisions = zeros(1,nImages);
recalls = zeros(1,nImages);
f_measures = zeros(1,nImages);
for i=1:nImages
    
    img = imread([imagePath '/' images(i).name]);
    label = imread([labelPath '/' labels(i).name]);

    im_data = img(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, acceptedSize, 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

    scores = net.forward({im_data});  
    scores = scores{1};

    predict = zeros(size(scores(:,:,2)));
    predict(find(scores(:,:,1)<=scores(:,:,2))) = 1;
    predict = predict';

%     imagesc(scores(:,:,1)')
    tmp = predict(find(predict == label));
    tmp = tmp(find(tmp == 1));
    if size(tmp,1) ~= 0
        TP = size(tmp,1);
    else
        TP = 0;
    end
    
    tmp = predict(find(predict ~= label));
    tmp = tmp(find(tmp == 1));
    if size(tmp,1) ~= 0
        FP = size(tmp,1);
    else
        FP = 0;
    end
    
    FN = sum(sum(label)) - TP;
    precisions(i) = double(TP/(TP+FP));
    if TP == 0 && FP == 0
        precisions(i) = 1;
    end
    recalls(i) = double(TP/(TP+FN));
    if TP == 0 && FN == 0
        recalls(i) = 1;
    end
    f_measures(i) = double((2*precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
    if precisions(i) == 0 && recalls(i) == 0
        f_measures(i) = 1;
    end
    
end
save precision_recall_fmeasure_bungalows.mat precisions recalls f_measures





