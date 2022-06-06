classdef ICG
    methods(Static)
        function [result] = parseEdgeMatcherFile(fileName)
            %%
            % Parse edge matcher result file and store numeric content
            % in the result return value.
            % @param fileName Edge matcher result file path.
            % Returns Array that contains egde matcher result values.
            result = [];
            fid = fopen(fileName, 'r');
            while ~feof(fid)
                textLine = fgetl(fid);
                
                if(ICG.isCommentOrEmpty(textLine))
                    continue;
                end
                
                content = sscanf(textLine, '%f', [1 4]);
                result = [result; content];
            end
            fclose(fid);
        end
        
        function [valid] = isCommentOrEmpty(textLine)
            %%
            % Check if the input text is empty or is a comment (first sign #).
            % @param textLine String text line that is checked.
            % Returns True if empty or a comment, otherwise false.
            valid = false;
            if(length(textLine) == 0)
                valid = true;
            end
            k = strfind(textLine, '#');
            
            if(length(k) > 0)
                valid = true;
            end
        end
        
        function [edgeResults, legendResults] = getEdgeMatcherResults(fileNames)
            %%
            % Extracts necessary information from the edge matcher results
            % file.
            % @param fileNames Array of result file names.
            % Returns important part of the edge results presented in the
            % file as array.
            results = {};
            edgeResults = [];
            legendResults = {};
            
            importantResultColumn = 4;
            
            for i = 1:length(fileNames)
                [filepath, name, ext] = fileparts(fileNames{i});
                legendResults{i} = strrep(name, '_', ' ');
                results{i, 1} = ICG.parseEdgeMatcherFile(fileNames{i});
                edgeCounts = results{i, 1}(:, importantResultColumn);
                edgeResults = [edgeResults, edgeCounts];
            end
        end
        
        function [] = displayEdgeMatcherResultChart(edgeResults, legendResults, plotTitle)
            %%
            % Displays determined edge matcher results in a chart.
            % @param edgeResuls Array that contains the results of the
            % edge matching process.
            % @param legend String legend entries, used for plot
            % description.
            meanResults = mean(edgeResults);
            stdDevResults = std(edgeResults);
            
            figure('Name', 'd>1');
            hold on;
            barChart = bar(edgeResults);
            title(strcat(plotTitle, ': ', 'd>1'), 'Interpreter', 'none');
            xlabel('# Frames', 'Interpreter', 'none');
            ylabel('# Counted Edges', 'Interpreter', 'none');
            legend(barChart, legendResults)
            
            figure('Name', 'Mean/StdDev');
            hold on;
            title(strcat(plotTitle, ': ', 'Mean/StdDev'), 'Interpreter', 'none');
            errorChart = errorbar(meanResults, stdDevResults, 'o', 'MarkerEdgeColor','red','MarkerFaceColor','red', 'Color', 'red');
            xlabel('Algorithm', 'Interpreter', 'none');
            ylabel('# Edges', 'Interpreter', 'none');
            xticks([1:length(meanResults)]);
            xtickangle(45);
            set(gca, 'XTickLabel', legendResults);
        end
        
        function [valid] = existsOrCreate(fileDir, varargin)
            %%
            % Checks if file directory exists if not user will be asked to
            % create.
            % param fileDir File directory to check and to create.
            % param [createWithoutPrompt] True to create directory without
            % prompt, otherwise user will be asked.
            % Returns true if directory exists otherwise false.
            valid = false;
            if(~exist(fileDir, 'dir'))
                createWithoutPrompt = false;
                if(nargin > 1)
                    createWithoutPrompt = varargin{1};
                end
                
                if(createWithoutPrompt)
                    mkdir(fileDir);
                    valid = true;
                else
                    fprintf('Directory "%s" does not exist.\n', fileDir)
                    reply = input('Do you want to create it? Y/N [Y]: ', 's');
                    
                    if(isempty(reply) | strcmpi(reply, 'y') == 1)
                        mkdir(fileDir);
                        valid = true;
                    end
                end
                
                
            else
                valid = true;
            end
        end
        
        function [rangeImg] = scaleLowHigh(img)
            %%
            % Rearange values usually taken for writing images to HDD,
            % similar to the brakets in imshow(img,[])
            rangeImg = (double(img) - min(img(:))) ./ (max(img(:)) - min(img(:)));
        end
        
        function [E] = nmsEdgeImage(filename, varargin)
            %%
            % Non Maximum Suppression of image file.
            %
            options.r = 1; % radius for nms supr
            options.s = 5; % radius for supr boundaries
            options.m = 1.01; % multiplier for conservative supr
            options.nThreads = 4; % number of threads for evaluation
            options.t = 0.2; % binary image threshold
            
            if ~cellfun('isempty', varargin)
                tmp = varargin{1};
                if isfield(tmp, 'r')
                    options.r = tmp.r;
                end
                if isfield(tmp, 's')
                    options.s = tmp.s;
                end
                if isfield(tmp, 'm')
                    options.m = tmp.m;
                end
                if isfield(tmp, 'nThreads')
                    options.nThreads = tmp.nThreads;
                end
                if isfield(tmp, 't')
                    options.t = tmp.t;
                end
                
            end
            
%             if isstring(filename) == 0
%                 E = imread(filename);
%             else
                E = filename;
%             end
            
            [w h c] = size(E);
            
            if(c > 1)
                error('No probability mask.')
            end
            E = single(E);
            
            maxVal = 1.0;
            
            if max(E(:)) > 1
                maxVal = 255.0;
            end
            
            E = E ./ maxVal;
            [Ox, Oy] = gradient2(convTri(E,4));
            [Oxx, ~] = gradient2(Ox);
            [Oxy, Oyy] = gradient2(Oy);
            O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx+1e-5)), pi);
            
            % E...original edge map
            % O...orientation map
            % r...radius for nms supr
            % s...radius for supr boundaries
            % m...multiplier for conservative supr
            % nThreads...number of threads for evaluation
            E = edgesNmsMex(E, O, options.r, options.s, options.m, options.nThreads);
            
            E = im2bw(E, options.t);
        end
        
        function [E] = edgeThinning(filename, varargin)
            %%
            % Edge thinning.
            options.P = 5; % remove pixel that are smaller than P.
            
            if ~cellfun('isempty', varargin)
                tmp = varargin{1};
                if isfield(tmp, 'P')
                    options.P = tmp.P;
                end
            end
            
%              if isstring(filename) == 0
%                  E = imread(filename);
%              else
                E = filename;
%              end
            
            [w h c] = size(E);
            
            if(c > 1 || max(E(:)) > 1)
                error('No binary image')
            end
            
            % remove objects smaller than P pixels
            E = bwareaopen(E, options.P);
            % Removes isolated pixels
            E = bwmorph(E, 'thin', Inf);
            E = bwmorph(E, 'clean');
        end
        
        function [RGB_Image] = convertBinImage2RGB(BinImage, channel, r, g, b)
            [fil, col] = size(BinImage);
            RGB_Image = ones(fil,col,3);
            [posX , posY] = find(BinImage==channel);
            numIter = size(posX,1)*size(posX,2);
            for ii = 1 : numIter
                RGB_Image(posX(ii),posY(ii), 1) = r;
                RGB_Image(posX(ii),posY(ii), 2) = g;
                RGB_Image(posX(ii),posY(ii), 3) = b;
            end % for
        end % function
        
        function [whiteBackground] = isBackgroundWhite(grayImg)
            [w h c] = size(grayImg);
            if(c > 1)
                error('No single channel image')
            end
            if(max(grayImg(:))==1)
                grayImg = grayImg*255;
            end
            whiteCounter = length((find(grayImg==255)));
            blackCounter = w*h - whiteCounter;
            whiteBackground = (whiteCounter > blackCounter);
        end
    end
end
