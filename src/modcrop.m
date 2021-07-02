function img = modcrop(img,q)
% modcrop - Crops an image so that the output M-by-N image satisfies mod(M,q)=0 and mod(N,q)=0
sz(1) = size(img,1);
sz(2) = size(img,2);
sz = sz - mod(sz,q);
img = img(1:sz(1), 1:sz(2),:);