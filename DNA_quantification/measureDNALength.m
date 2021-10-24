%% Script to quantify DNA content on segmented TEM images
%
% author: Fabienne K. Roessler
% date: July 2018
% --------------------------------------------------------

% ATTENTION! 
% TO DO: Define path to folder containing segmented images
inputpath = ...;
    
% Constants
pixelsize = 2.0; %in nm
fac = (1+sqrt(2))/2;

% List of tile sets saved in input folder
squares = dir(inputpath);

% Create empty arrays for saving names of tile sets and length of
% measured DNA content
nameSquares = string(zeros(0, 1));
totalLengthsSquares = double(zeros(length(squares)-2, 1));

numSquares = 1;

% Loop over different tile sets
for i = 3:length(squares)
    display(['Measure DNA length of tile set ',squares(i).name]);
    
    % Save name of tile set
    nameSquares(i-2, 1) = squares(i).name;

    currentFolder = strcat(inputpath, squares(i).name, '\');
    % Get list of images from current tile set
    images = dir(currentFolder);
    
    finalLengthsArray = double(zeros(length(images)-2, 1));
    count = 1; 
    
    % Loop over images
    for im = 3:length(images)
        
        % Read segmented image
        m = imread(strcat(currentFolder, images(im).name));
        mask = m < 1; %Pixels of Value 2 are true
        
        % Close mask (morphologically --> dilation followed by erosion)
        closedMask = bwmorph(mask, 'close');
        
        % Skeletonize mask
        skelMask = bwmorph(closedMask,'thin',Inf);
        
        % Filter skeletonized mask by length
        skelMask_filtered = bwareaopen(skelMask, 55);
        
        % Remove small branches from filtered skeletonized mask
        skelMask_filtered_debranched = bwmorph(skelMask_filtered,'spur', 15);
        
        % Count DNA pixels of skeletonized mask
        DNA_pixels = sum(sum(skelMask_filtered_debranched));
        
        % Determine total DNA length (in nm) on one image
        finalLength = DNA_pixels*fac*pixelsize;
        
        % Save total length in array
        finalLengthsArray(count,1) = finalLength;
        count = count + 1;
    end
    
    % Sum up DNA content determined on images of one tile set
    totalLengthsSquares(numSquares,1) = sum(finalLengthsArray);
    
    numSquares = numSquares + 1;
end

T = table(nameSquares, totalLengthsSquares)

% Save results to Excel file
titleExcelSheet = ["Square", "Total DNA length [nm]"];
xlswrite(strcat(inputpath, 'resultsDNALength.xlsx'), titleExcelSheet);
xlswrite(strcat(inputpath, 'resultsDNALength.xlsx'),table2array(T), 1, 'A2');