%% Integrated pre-processing method for multitemporal very high-resolution satellite images written by Taeheon, Kim (2020.01.14)
clc;clear;close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import image and metadata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd('E:\Taeheon\SURF_region growing\Tiff');
[f1,path] = uigetfile({'*.tif; *.jpg; *.bmp; *.gif'}, 'Please Select The Reference Image');
if path == 0
    return;
end
cd('E:\Taeheon\SURF_region growing\Tiff');
[f2,path] = uigetfile({'*.tif; *.jpg; *.bmp; *.gif'}, 'Please Select The Sensed Image)');
if path == 0
    return;
end
image1=imread(f1);
image2=imread(f2);
image1=image1(:,:,1:4); 
image2=image2(:,:,1:4); 
image1=double(image1);
info1 = imfinfo(f1);
info2 = imfinfo(f2); 
image2=double(image2);
[row, col, band] = size(image1);
Ref = double(image1(:,:,3));
Ref = Ref - min(Ref(:));
Sub = double(image2(:,:,3));
Sub = Sub - min(Sub(:));
image1_vi=zeros(size(image1));
image2_vi=zeros(size(image2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% image conversion (14bits -> 8bits) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KOMPSAT-3A
parfor m=1:band
    vi_img1=image1(:,:,m);
    uni_img1=sort(vi_img1(:));
    image1_min=uni_img1(round(length(uni_img1)*(1/100)));
    image1_max=uni_img1(round(length(uni_img1)*(99/100)));
    image1_view(:,:,m)=((image1(:,:,m)-image1_min)./(image1_max-image1_min))*255;
    vi_img2=image2(:,:,m);
    uni_img2=sort(vi_img2(:));
    image2_min=uni_img2(round(length(uni_img2)*(1/100)));
    image2_max=uni_img2(round(length(uni_img2)*(99/100)));
    image2_view(:,:,m)=((image2(:,:,m)-image2_min)./(image2_max-image2_min))*255;
end
image1_view=uint8(image1_view);
image2_view=uint8(image2_view);
Ref_use=image1_view(:,:,3);
Sub_use=image2_view(:,:,3);
Ref = double(Ref/max(max(Ref)));
Sub = double(Sub/max(max(Sub)));

% For image registration
Ref_use = uint8(255*Ref); Ref_use=double(Ref_use);
Sub_use = uint8(255*Sub); Sub_use=double(Sub_use);
image1=uint16(image1);
image2=uint16(image2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Std-Mean Linear Stretch 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ref_mean=mean(mean(Ref_use));
Ref_std=std(Ref_use(:));
Ref_max=Ref_mean+Ref_std;
Ref_min=Ref_mean-Ref_std;
Ref_k=((Ref_use-Ref_min)./(Ref_max-Ref_min))*255;
Ref_use=uint8(Ref_k);Ref_use=gather(Ref_use);
Sub_mean=mean(mean(Sub_use));
Sub_std=std(Sub_use(:));
Sub_max=Sub_mean+Sub_std;
Sub_min=Sub_mean-Sub_std;
Sub_k=((Sub_use-Sub_min)./(Sub_max-Sub_min))*255;
Sub_use=uint8(Sub_k);Sub_use=gather(Sub_use);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CPs extraction using SURF method %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find the SURF features
points1 = detectSURFFeatures(Ref_use);
points2 = detectSURFFeatures(Sub_use);
% extract the features
[f1, vpts1] = extractFeatures(Ref_use, points1,'Upright',true);
[f2, vpts2] = extractFeatures(Sub_use, points2,'Upright',true);
% find the matchedfeatures
[indexPairs, matchmetric] = matchFeatures(f1, f2, 'Method','Approximate','Metric', 'SSD', 'MaxRatio',0.6); 

matchedRefPoints1 = vpts1(indexPairs(:, 1));
matchedSubPoints2 = vpts2(indexPairs(:, 2));
Ref=gather(Ref);Sub=gather(Sub);
figure; showMatchedFeatures(Ref,Sub,matchedRefPoints1,matchedSubPoints2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Outliers removal by iterative affine Transformation estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TH_R = 5;
matchedRefPoints1=double(matchedRefPoints1.Location);
matchedSubPoints2=double(matchedSubPoints2.Location);
[matchedSubPoints_final, matchedRefPoints_final] = outlier_removal_iter_affine(matchedSubPoints2,matchedRefPoints1, TH_R);
[matchedSubPoints_final, matchedRefPoints_final] = PreProcessCp2tform(matchedSubPoints_final, matchedRefPoints_final);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Warping sensed image using improved piecewise linear transformation (Remote sensing written by Han)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% image-to-image registration using affine 
[CPs_IPL_r,CPs_IPL_s] = IPL(image2,matchedRefPoints_final,matchedSubPoints_final);
tform_IPL = cp2tform(CPs_IPL_s, CPs_IPL_r,'piecewise');
[registered_IPL,RB] = imtransform(image2(:,:,1:4), tform_IPL,'nearest', 'FillValues', 255,'XData', [1 size(image2,2)],'YData', [1 size(image2,1)]);
registered_IPL=double(registered_IPL);
registered=zeros(size(registered_IPL));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate NDVI Mask 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
registered_IPL=double(registered_IPL);
image1=double(image1);
NDVI_Tar=(registered_IPL(:,:,4)-registered_IPL(:,:,3))./(registered_IPL(:,:,4)+registered_IPL(:,:,3));
NDVI_Ref=(image1(:,:,4)-image1(:,:,3))./(image1(:,:,4)+image1(:,:,3));
NDVI_Tarth=graythresh(NDVI_Tar);NDVI_Refth=graythresh(NDVI_Ref);
ND_bin = zeros(size(NDVI_Tar));
ND_bin(NDVI_Tar<NDVI_Tarth&NDVI_Ref<NDVI_Refth) = 1; 
ND_bin = medfilt2(ND_bin, [9 9]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extraction IPS(Invariant Pixel Samples) using Z-score and Region Growing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Extraction PIFs using Z-score %%%%%%%%%%%%
% find keypoints location
registered_IPL=double(registered_IPL(:,:,1:4));
image1=double(image1(:,:,1:4));
band=4;
matching_points=uint16(matchedRefPoints_final);

Z=zeros(size(image1));
Z_score=zeros(size(image1));
Z_score_value=zeros(length(matching_points),1);

% PIFs difference brightness value
for b=1:band
    Z(:,:,b)=image1(:,:,b)-registered_IPL(:,:,b);
end
for b=1:band
 value=Z(:,:,b);
 Z_std=std(value(:));
 Z_mean=mean(value(:));
 Z_score(:,:,b)=(Z(:,:,b)-Z_mean)./(Z_std);
end

% Band 1 ~ Band 4 
Z_score=sqrt(Z_score(:,:,1).^2+Z_score(:,:,2).^2+Z_score(:,:,3).^2+Z_score(:,:,4).^2); 

ind=1;
for i=1:length(matching_points)
    if(matching_points(i)~=0)
        if(ND_bin(matching_points(i,2),matching_points(i,1))==1)
            PIF_X(ind,1)=matching_points(i,1);
            PIF_Y(ind,1)=matching_points(i,2);
            ind=ind+1;
        end
    end
        
end
img_PIFs=[PIF_X(:),PIF_Y(:)];

for i = 1 : length(img_PIFs(:,2))
Z_score_data(i)=Z_score(PIF_Y(i),PIF_X(i));
end
%%%%%%%%%%%% Extraction IPS using Region_Growing %%%%%%%%%%%%
% PIFs: Seed pixels
% image=Z-score image
[row3,col3]=size(img_PIFs);
IPS_loc=zeros(size(Z_score));
for i=1:row3
    if(img_PIFs(i,2)~=0&&img_PIFs(i,1)~=0)
        IPS=regiongrowing_ori(Z_score,img_PIFs(i,2),img_PIFs(i,1),0.2);
    IPS=double(IPS);
    IPS_loc=IPS_loc+IPS;
    end
        
end
IPS_final=IPS_loc;
IPS_final(IPS_final>=1)=1;
[loc_y,loc_x]=find(IPS_final>=1);

for i = 1 : length(loc_y)
PIFs_data(i)=Z_score(loc_y(i),loc_x(i)); %Sub
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test_data and IPS %% (Traning data: 70%, Test data: 30%)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num=length(loc_y)
idx = randperm(num); idx=idx(:);
idx_training = idx>num*0.3;
idx_test = idx<=num*0.3;
training_y=loc_y.*idx_training; training_y(training_y==0)=[];
training_x=loc_x.*idx_training; training_x(training_x==0)=[];
test_y=loc_y.*idx_test; test_y(test_y==0)=[]; 
test_x=loc_x.*idx_test; test_x(test_x==0)=[]; 

Ref_traning_data=zeros(length(training_y),band);
Sub_traning_data=zeros(length(training_y),band);
Ref_traning_CPs=zeros(length(PIF_Y),band);
Sub_traning_CPs=zeros(length(PIF_Y),band);

for b=1:band
    for i = 1 : length(training_y)
Ref_traning_data(i,b)=image1(training_y(i),training_x(i),b); %Sub
Sub_traning_data(i,b)=registered_IPL(training_y(i),training_x(i),b); %Ref
    end
end
for b=1:band
    for i = 1 : length(PIF_Y)
Ref_traning_CPs(i,b)=image1(PIF_Y(i),PIF_X(i),b); %Sub
Sub_traning_CPs(i,b)=registered_IPL(PIF_Y(i),PIF_X(i),b); %Ref
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalize registered image using IPS and linear regression %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proposed method
registered_IPL=double(registered_IPL);
band=4;
for i=1:band
X_IPS=Sub_traning_data(:,i);
Y_IPS=Ref_traning_data(:,i);
PIFa_mean=mean(X_IPS(:));
PIFb_mean=mean(Y_IPS(:));
PIFa_std=std(X_IPS(:));
PIFb_std=std(Y_IPS(:));
ak=PIFb_std/PIFa_std
bk=PIFb_mean-ak*PIFa_mean
img_nor_filter(:,:,i)=ak*registered_IPL(:,:,i)+bk;

YF=ak*registered_IPL+bk;
heatscatter(X_IPS(:),Y_IPS(:),'100','30','.','1','1','Geo-rectified sensed image','Reference image');
end
output_image = uint16(img_nor_filter);
