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
labels = dir([labelPath '/' imageType]);
nImages = length(images);

for i=1:nImages
    
    img = imread([imagePath '/' images(i).name]);
    
    im_data = img(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, acceptedSize, 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

    scores = net.forward({im_data});  
    scores = scores{1};

    predict = zeros(size(scores(:,:,2)));
    predict(find(scores(:,:,1)<=scores(:,:,2))) = 255;
	predict = predict';
    imwrite(predict, ['./predictedLabels_bungalows/', 'predict_', images(i).name]);

%     imagesc(scores(:,:,1)')

end

