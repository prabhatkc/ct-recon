function plotLayers(im)
%
% plotLayers(im) make subplots where the k'th subplot is the
% layer im(:,:,k) displayed as an image

% Number of layers to show
numLayers = size(im,3);

% Determine horizontal and vertical number of subplots
s1 = floor(sqrt(numLayers));
s2 = s1 + 2;

% Determine the overall minimum an maximum entries in im for setting common
% color axis
imVec  = im(:);
maxVal = max(imVec);
minVal = min(imVec);
ca     = [minVal,maxVal];

% Loop through the layers of im, make new subplot, display current layer as
% image, set color axis and title.
figure;
for k = 1:numLayers
    subplot(s1,s2,k)
    imagesc(im(:,:,k))
    axis image off
    colormap(gray); %ject
    caxis(ca)
    title(['( *, *, ',num2str(k),')'], 'FontSize', 9)
end