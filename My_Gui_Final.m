function varargout = My_Gui_Final(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @My_Gui_Final_OpeningFcn, ...
                   'gui_OutputFcn',  @My_Gui_Final_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before My_Gui_Final is made visible.
function My_Gui_Final_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

handles.sai = 1;
handles.sai1= 1;
handles.var = 1;


axes(handles.axes3);
imshow('Logo.png');

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes My_Gui_Final wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = My_Gui_Final_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;


% --- Executes on selection change in leo1_popup.
function leo1_popup_Callback(hObject, eventdata, handles)

val = get(handles.leo1_popup, 'Value');
 
switch val;
    
    case 1 
        str = 'Load Image'; 

    case 2  
        str = 'Load Video';
        
    case 3
        str = 'Live cam';

end
 set(findobj('tag', 'load_push'), 'String', str);
        
        
handles.var = val;
        
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function leo1_popup_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in leo2_popup.
function leo2_popup_Callback(hObject, eventdata, handles)




for k = 1:handles.numFrames-1

    if(handles.var == 2)
    
    mov(k).cdata = handles.vidFrames(:,:,:,k);
    mov(k).colormap = [];
    
    I = mov(k).cdata;
    
    elseif (handles.var == 1)
     
        I = handles.I;
    end
        
val = get(handles.leo2_popup, 'Value');
 
switch val;
    
    case 1  % Actions
        
    case 2  % Gray Color Space
         Im = I;
         gr = rgb2gray(Im);
         axes(handles.axes1)
         imshow(I)
         axes(handles.axes2)
         imshow(gr)
         colormap gray
        
    case 3 % HSV Color Space
        
         M = I;
         H = rgb2hsv(M);
         axes(handles.axes1)
         imshow(I)
         axes(handles.axes2);
         imshow(H)
        
    case 4 % YCbCr color Space
         M = I;
         H = rgb2ycbcr(M);
         axes(handles.axes1)
         imshow(I)
         axes(handles.axes2);
         imshow(H)
         
end
          

end
guidata(hObject, handles);




% --- Executes during object creation, after setting all properties.
function leo2_popup_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in leo3_popup.
function leo3_popup_Callback(hObject, eventdata, handles)


for k = 1:handles.numFrames-1

    if(handles.var == 2)
    
    mov(k).cdata = handles.vidFrames(:,:,:,k);
    mov(k).colormap = [];
    
    I = mov(k).cdata;
    
    elseif (handles.var == 1)
     
        I = handles.I;
    end
        
val = get(handles.leo3_popup, 'Value');
 
 switch val;
    
    case 1  % Actions
        
    case 2  % Computes Histogram
         Im = I;
         origimage = Im(:,:,1);
         i = imhist(origimage);
         axes(handles.axes1);
         imshow(I);
         axes(handles.axes2);
         plot(i);
          
    case 3  % Equalizes the Histogram
         Im = I;
         I1 = Im(:,:,1);
         i = histeq(I1);
         I2 = imhist(i);
         axes(handles.axes1);
         imshow(I);
         axes(handles.axes2);
         plot(I2);
        
        
 end
end
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function leo3_popup_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in leo4_popup.
function leo4_popup_Callback(hObject, eventdata, handles)


for k = 1:handles.numFrames-1

    if(handles.var == 2)
    
    mov(k).cdata = handles.vidFrames(:,:,:,k);
    mov(k).colormap = [];
    
    I = mov(k).cdata;
    
    elseif (handles.var == 1)
     
        I = handles.I;
    end
        
val = get(handles.leo4_popup, 'Value');
 
switch val;
    
    case 1  % Actions
        
    case 2  %  Morphological operations   %Dilate
         Im = I ; % Select sk.png Image to see results clearly   
         Im(4:6,4:7) = 1;
         SE = strel('square',double(ceil(handles.sai))); % Changing the dilate percentage 
         BW2 = imdilate(Im,SE);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(BW2);
    
    case 3     %%% Erode
         Im = I ;  % Select sk.png Image to see results clearly   
         SE = strel('arbitrary',double(ceil(handles.sai))); % Changing the Erode percentage
         BW2 = imerode(Im,SE);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(BW2);
          
    case 4  % Open
         Im = I ;  %  Select Snowflakes Image to see results clearly   
         se = strel('disk',double(ceil(handles.sai))); % Changing the Open percentage
         afterOpening = imopen(Im,se);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(afterOpening,[]);
  
    case 5  % Close
         Im = I ; %Select cir.jpeg Image to see results clearly  
         se = strel('disk',double(ceil(handles.sai))); % Changing the close percentage
         closeBW = imclose(Im,se);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(closeBW);
      
end

end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function leo4_popup_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in leo5_popup.
function leo5_popup_Callback(hObject, eventdata, handles)


for k = 1:handles.numFrames

    if(handles.var == 2)
    
    mov(k).cdata = handles.vidFrames(:,:,:,k);
    mov(k).colormap = [];
    
    I = mov(k).cdata;
    
    elseif (handles.var == 1)
     
        I = handles.I;
    end
        
val = get(handles.leo5_popup, 'Value');
 
switch val;
    
    case 1  % Actions

    case 2  % Kernal Blur
         origimage = I;
         origimage = origimage(:,:,1);
         H = fspecial('disk',double(ceil(handles.sai1))); % Changing the Blur percentage
         blurred = imfilter(origimage,H,'replicate');
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(blurred);
         
    case 3  % Motion Blur
         Im = I ;
         h = fspecial('motion',double(ceil(handles.sai1)),double(ceil(handles.sai1))); % Changing the Blur percentage
         boundaryReplicateRGB = imfilter(Im,h,'replicate'); 
         axes(handles.axes1)
         imshow(I); 
         axes(handles.axes2)
         imshow(boundaryReplicateRGB)
 
    case 4 % sobel operator
         im = I ;
         I1 = im(:,:,1);
         sob_im = edge(I1,'sobel');
         axes(handles.axes1)
         imshow(I); 
         axes(handles.axes2)
         imshow(sob_im);
         colormap gray
        
    case 5  % Laplacian operator
         Im = I ;
         I1 = Im(:,:,1);
         H = fspecial('laplacian');
         blurred = imfilter(I1,H);
         axes(handles.axes1)
         imshow(I); 
         axes(handles.axes2)
         imshow (blurred)
         
    case 6  % Sharpen Image
         Im = I;
         I1 = Im(:,:,1);
         H = fspecial('unsharp');
         sharpened = imfilter(I1,H,'replicate');
         axes(handles.axes1)
         imshow(I); 
         axes(handles.axes2)
         imshow(sharpened); 
       
    case 7  % Edge detection using Canny
         I1 = I ;
         img = I1(:,:,1);
         Im = edge(img,'canny');
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(Im)
      
        
    case 8  % Extraction of lines using Hough Transform
         Im = I ;
         I = Im(:,:,1);
         rotI = imrotate(I,double(ceil(handles.sai1)),'crop');% Changing the rotation percentage
         BW = edge(rotI,'canny');
         [H,theta,rho] = hough(BW);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(rotI), hold on
         P = houghpeaks(H,double(ceil(handles.sai1)),'threshold',double(ceil(handles.sai1))); % Changing the threshold percentage
         x = theta(P(:,2));
         y = rho(P(:,1));
         plot(x,y,'s','color','black');
         lines = houghlines(BW,theta,rho,P,'FillGap',5,'MinLength',7);
         imshow(rotI), hold on
         max_len = 0;
         for k = 1:length(lines)
         xy = [lines(k).point1; lines(k).point2];
         plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

       % Plot beginnings and ends of lines
         plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
         plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

       % Determine the endpoints of the longest line segment
         len = norm(lines(k).point1 - lines(k).point2);
           if ( len > max_len)
              max_len = len;
              xy_long = xy;
           end
         end
  
     % highlight the longest line segment
       plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');
         
          
        
    case 9   % Extraction of circles using Hough Transform
         A = I ;  % Select Coins Image to see results clearly
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(A)
         [centers, radii, metric] = imfindcircles(A,[15 30]);
         centersStrong5 = centers(:,:);
         radiiStrong5 = radii(:,:);
         metricStrong5 = metric(:,:);
         viscircles(centersStrong5, radiiStrong5,'EdgeColor','b');
         
    case 10  % Contour
         im = I ;
         axes(handles.axes1)
         imshow(I)
         Im = im(:,:,1);
         I = im2double(Im);
         axes(handles.axes2)
         imshow(I);
         isocontour(I,0.5);
        
         
    case 11  % Harris and non-maximal suppression
         Im = I ; % Select Rice Image to see results clearly
         axes(handles.axes1)
         imshow(Im);
         im = Im(:,:,1);
         sigma = double(ceil(handles.sai1));% Changing the sigma, threshold, radius percentage
         thresh = double(ceil(handles.sai1));
         radius = double(ceil(handles.sai1));
         disp = 1;
         axes(handles.axes2)
         imshow(I);
         harris(im, sigma, thresh, radius, disp);
             
    case 12  % Extraction of Features using FAST 
         Im = I ;
         im = im2single(Im);
         I = im(:,:,1);
         cornerDetector = vision.CornerDetector('Method','Local intensity comparison (Rosten & Drummond)');
         %Find corners
          pts = step(cornerDetector, I);
         %Create the markers. The color data range must match the data range of the input image. The color format for the marker is [red, green, blue].
          color = [1 0 0];
          drawMarkers = vision.MarkerInserter('Shape', 'Circle', 'BorderColor', 'Custom', 'CustomBorderColor', color);
         %Convert the grayscale input image I to an RGB image J before inserting color markers.
          J = repmat(I,[1 1 3]);
          J = step(drawMarkers, J, pts);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(J);
          
    case 13  % Extraction of Features using SURF
         im = I;
         I = im(:,:,1);
         points = detectSURFFeatures(I);
         [features, valid_points] = extractFeatures(I, points);
         
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(I); hold on;
         plot(valid_points.selectStrongest(5),'showOrientation',true);
       
    case 14    % Matching two images with different views
         original = I ;
         text(size(original,2),size(original,1)+15, ...
                   'Image courtesy of Massachusetts Institute of Technology', ...
                     'FontSize',7,'HorizontalAlignment','right');
         scale = 0.7;
         J = imresize(original, scale); 

         theta = double(ceil(handles.sai1));                            % Changing the rotation percentage
         distorted = imrotate(J,theta); 
         ptsOriginal  = detectSURFFeatures(original);
         ptsDistorted = detectSURFFeatures(distorted);
         [featuresIn   validPtsIn]  = extractFeatures(original,  ptsOriginal);
         [featuresOut validPtsOut]  = extractFeatures(distorted, ptsDistorted);
         index_pairs = matchFeatures(featuresIn, featuresOut);
         matchedOriginal  = validPtsIn(index_pairs(:,1));
         matchedDistorted = validPtsOut(index_pairs(:,2));
%          cvexShowMatches(original,distorted,matchedOriginal,matchedDistorted);
%          title('Putatively matched points (including outliers)');

         geoTransformEst = vision.GeometricTransformEstimator; % defaults to RANSAC

         % Configure the System object.
          geoTransformEst.Transform = 'Nonreflective similarity';
          geoTransformEst.NumRandomSamplingsMethod = 'Desired confidence';
          geoTransformEst.MaximumRandomSamples = 1000;
          geoTransformEst.DesiredConfidence = 99.8;

         [tform_matrix inlierIdx] = step(geoTransformEst, matchedDistorted.Location, ...
                                              matchedOriginal.Location);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         cvexShowMatches(original,distorted,matchedOriginal(inlierIdx),...
                             matchedDistorted(inlierIdx),'ptsOriginal','ptsDistorted');
         title('Matching points (inliers only)');

         tform_matrix = cat(2,tform_matrix,[0 0 1]'); % pad the matrix
         Tinv  = inv(tform_matrix);

         ss = Tinv(2,1);
         sc = Tinv(1,1);
         scale_recovered = sqrt(ss*ss + sc*sc);
         theta_recovered = atan2(ss,sc)*180/pi;

         t = maketform('affine', double(tform_matrix));
         D = size(original);
         recovered = imtransform(distorted,t,'XData',[1 D(2)],'YData',[1 D(1)]);

         axes(handles.axes2)
         imshowpair(original,recovered,'montage')
        
    case 15  % Fundamental Matrix
         Im = I ;
         points1 = detectSURFFeatures(Im);
         scale = 0.7;
         J = imresize(Im, scale);
         theta = 2;
         distorted = imrotate(J,theta);
         points2 = detectSURFFeatures(distorted);

        % Display 10 strongest points in an image and on command line
    
         strongest1 = points1.selectStrongest(8);
         strongest2 = points2.selectStrongest(8);
%          figure,imshow(I); hold on;
%          plot(strongest1);   % show location and scale
         A1 = strongest1.Location;
%          figure,imshow(distorted); hold on;
%          plot(strongest2);   % show location and scale
         A2 = strongest2.Location;
         fNorm8Point = estimateFundamentalMatrix(A1, A2, ...
                                'Method', 'Norm8Point');
         A = zeros(8,8);
         
         for i = 1:8
    
                  A(i,:) = [A1(i,1)*A2(i,1), ...
                             A1(i,1)*A2(i,2), ...
                              A1(i,1), ...
                               A1(i,2)*A2(i,1), ...
                                A1(i,2)*A2(i,2), ...
                                 A1(i,2), ...
                                  A2(i,1) , ...
                                   A2(i,2)];
         end
                

         B = [-1;-1;-1;-1;-1;-1;-1;-1];
         F = inv(A'*A)*A'*B ;
         F1 = [F(1) F(2) F(3);F(4) F(5) F(6); F(7) F(8) 1];
         display('Fundamental Matrix Using Function');    
         display(fNorm8Point);
         display('Fundamental Matrix Using Manually');
         display(F1);
         
         
         
         
    case 16  % Bounding Box
         I =I;
         Im = im2bw(I);
         axes(handles.axes1)
         imshow(I);
         axes(handles.axes2)
         imshow(I); hold on
         Im = imfill(Im,'holes');
         L = bwlabel(Im);
         S = regionprops(L,'BoundingBox');
        for i=1:length(S)
           rectangle('position',S(i).BoundingBox,'EdgeColor', 'blue');
        end
        hold off;
        
    case 17  % Mosaic
         im1=imread('im1.png');
         im2=imread('im2.png');
         im1f=figure; imshow(im1);
         im2f=figure; imshow(im2);

         figure(im1f), [x1,y1]=getpts;
         figure(im2f), [x2,y2]=getpts;
         figure(im1f), hold on, plot(x1,y1,'or');
         figure(im2f), hold on, plot(x2,y2,'or');

         T=maketform('projective',[x2 y2],[x1 y1]);
         T.tdata.T

         [im2t,xdataim2t,ydataim2t]=imtransform(im2,T);
         % now xdataim2t and ydataim2t store the bounds of the transformed im2
         xdataout=[min(1,xdataim2t(1)) max(size(im1,2),xdataim2t(2))];
         ydataout=[min(1,ydataim2t(1)) max(size(im1,1),ydataim2t(2))];
         % let's transform both images with the computed xdata and ydata
         im2t=imtransform(im2,T,'XData',xdataout,'YData',ydataout);
         im1t=imtransform(im1,maketform('affine',eye(3)),'XData',xdataout,'YData',ydataout);

         ims=im1t/2+im2t/2;
         figure, imshow(ims)

         imd=uint8(abs(double(im1t)-double(im2t)));
         % the casts necessary due to the images' data types
         imshow(imd);
         ims=max(im1t,im2t);
         axes(handles.axes2)
         imshow(ims);
end

end



% --- Executes during object creation, after setting all properties.
function leo5_popup_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in load_push.
function load_push_Callback(hObject, eventdata, handles)



% Open an image.
[filename,pathname]=uigetfile('*','open');

% whether you open an image.
if isequal(filename,0) 
    disp('User selected Cancel.') 
else
    disp(['User selected ', fullfile(pathname, filename), '.'])
end

full_file = fullfile(pathname,filename);



if (handles.var == 2)
  readerobj = VideoReader(full_file);

  handles.vidFrames = read(readerobj);
 handles.numFrames = get(readerobj, 'NumberOfFrames');

elseif (handles.var == 1)

  handles.I = imread(full_file);
  handles.numFrames = 2;

elseif (handles.var ==3)
    axes(handles.axes1)
    vid = videoinput('macvideo');
    preview(vid);
       
end

guidata(hObject, handles);
 

% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)

    handles.sai = get(hObject, 'value');
    
guidata(hObject, handles);
leo4_popup_Callback(hObject, eventdata, handles);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function edit1_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)

handles.sai1 = get(hObject, 'value');
    
guidata(hObject, handles);
leo5_popup_Callback(hObject, eventdata, handles);

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in calib_push.
function calib_push_Callback(hObject, eventdata, handles)

run calib_gui_normal

% --- Executes on button press in mosaic_push.
function mosaic_push_Callback(hObject, eventdata, handles)



% Open an image.
[filename,pathname]=uigetfile('*','open');

% whether you open an image.
if isequal(filename,0) 
    disp('User selected Cancel.') 
else
    disp(['User selected ', fullfile(pathname, filename), '.'])
end

full_file = fullfile(pathname,filename);

featWinLen   = 9;            % Length of feature window
maxNumPoints = int32(75);    % Maximum number of points
sizePano     = [400 680];
origPano     = [5 60];
classToUse   = 'single';

hsrc = vision.VideoFileReader(full_file, 'ImageColorSpace', ...
    'RGB', 'PlayCount', 1);

hcsc = vision.ColorSpaceConverter('Conversion', 'RGB to intensity');

 hcornerdet = vision.CornerDetector( ...
    'Method', 'Local intensity comparison (Rosen & Drummond)', ...
    'IntensityThreshold', 0.1, 'MaximumCornerCount', maxNumPoints, ...
    'CornerThreshold', 0.001, 'NeighborhoodSize', [21 21]);

hestgeotform = vision.GeometricTransformEstimator;

hgeotrans = vision.GeometricTransformer( ...
    'OutputImagePositionSource', 'Property', 'ROIInputPort', true, ...
    'OutputImagePosition', [-origPano fliplr(sizePano)]);

halphablender = vision.AlphaBlender( ...
    'Operation', 'Binary mask', 'MaskSource', 'Input port');

hdrawmarkers = vision.MarkerInserter('Shape', 'Circle', ...
    'BorderColor', 'Custom', 'CustomBorderColor', [1 0 0]);

hVideo1 = vision.VideoPlayer('Name', 'Corners');
hVideo1.Position(1) = hVideo1.Position(1) - 350;

hVideo2 = vision.VideoPlayer('Name', 'Mosaic');
hVideo2.Position(1) = hVideo1.Position(1) + 400;
hVideo2.Position([3 4]) = [750 500];

points   = zeros([0 2], classToUse);
features = zeros([0 featWinLen^2], classToUse);

while ~isDone(hsrc)
    % Save the points and features computed from the previous image
    pointsPrev   = points;
    featuresPrev = features;

    % To speed up mosaicking, select and process every 5th image
    for i = 1:5
        rgb = step(hsrc);
        if isDone(hsrc)
            break;
        end
    end
    I = step(hcsc, rgb);
    roi = int32([2 2 size(I, 2)-2 size(I, 1)-2]);

    % Detect corners in the image
    cornerPoints = step(hcornerdet, I);
    cornerPoints = cast(cornerPoints, classToUse);

    % Extract the features for the corners
    [features, points] = extractFeatures(I, ...
        cornerPoints, 'BlockSize', featWinLen);

    % Match features computed from the current and the previous images
    indexPairs = matchFeatures(features, featuresPrev);

    % Check if there are enough corresponding points in the current and the
    % previous images
    isMatching = false;
    if size(indexPairs, 1) > 2
        matchedPoints     = points(indexPairs(:, 1), :);
        matchedPointsPrev = pointsPrev(indexPairs(:, 2), :);

        % Find corresponding points in the current and the previous images,
        % and compute a geometric transformation from the corresponding
        % points
        [tform, inlier] = step(hestgeotform, matchedPoints, matchedPointsPrev);

        if sum(inlier) >= 4
            % If there are at least 4 corresponding points, we declare the
            % current and the previous images matching
            isMatching = true;
        end
    end

    if isMatching
        % If the current image matches with the previous one, compute the
        % transformation for mapping the current image onto the mosaic
        % image
        xtform = xtform * [tform, [0 0 1]'];
    else
        % If the current image does not match the previous one, reset the
        % transformation and the mosaic image
        xtform = eye(3, classToUse);
        mosaic = zeros([sizePano,3], classToUse);
    end

    % Display the current image and the corner points
    cornerImage = step(hdrawmarkers, rgb, cornerPoints);
    step(hVideo1, cornerImage);
    
    % Warp the current image onto the mosaic image
    transformedImage = step(hgeotrans, rgb, xtform, roi);
    mosaic = step(halphablender, mosaic, transformedImage, ...
        transformedImage(:,:,1)>0);
    step(hVideo2, mosaic);

end

release(hsrc);


% --- Executes on button press in homo_push.
function homo_push_Callback(hObject, eventdata, handles)

 
[filename1,pathname1]=uigetfile('*','open');
[filename2,pathname2]=uigetfile('*','open');

i1 = imread([pathname1 '/' filename1]);
i2 = imread([pathname2 '/' filename2]);

 Im1 = i1;
 Im2 = i2;
 
 i1 = Im1(:,:,1);
 i2 = Im2(:,:,1);

points1 = detectSURFFeatures(i1);
points2 = detectSURFFeatures(i2);

[f1 vpts1] = extractFeatures(i1, points1);
[f2 vpts2] = extractFeatures(i2, points2);

indexPairs = matchFeatures(f1,f2);
matched_pts1 = vpts1(indexPairs(:,1));
matched_pts2 = vpts2(indexPairs(:,2));
gte = vision.GeometricTransformEstimator;
gte.Transform = 'Projective';
[tform inlierIdx] = step(gte, matched_pts2.Location, matched_pts1.Location);
disp('Homography Matrix');
disp(tform);

% homography( i1, i2)

  