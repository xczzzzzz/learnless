%% Maximum Intensity Projection with color coding for depth information
% written by 
% Job Bouwman
% 02-12-2014
% contact: jgbouwman@hotmail.com

% input 3D image, the intensity projection is taken along the 3rd dimension
%matlab ��ֱ������M�ļ��е�һ����������Ҫ���亯�������ļ���һ��

function [maxtric_W]=colorMIP(image_3D)


%     %% crop for nice visualization: 
%     %�ü��Ի�����õ��Ӿ�Ч����
%     msk = round(image_3D./(image_3D + eps));
%     %����һ�α任������Ϊʲô�����任��roundΪȡ��
%     %epsΪ[2.22044604925031e-16]
% 
%     % The original indices in which the ROI is located: 
%     %��ʼ��
%     x1 = find(squeeze(sum(sum(msk, 1),3))); 
% %     t1=sum(msk, 1)
% %     t2=sum(sum(msk, 1),3)
% %     t3=squeeze(sum(sum(msk, 1),3))
%     y1 = find(squeeze(sum(sum(msk, 2),3)));
%     z1 = find(squeeze(sum(sum(msk, 1),2))); 
%     %�ҳ�����ͼƬ�а�����Ϣ��ά�ȣ��޳�����ͼƬ�а��У������ֵΪ0������
%     %squeeze����ժ����һά�ȵ������������ڽ�ά�����ã����ڱ����ж���2ά����
%     %��˼���֮��û��ʲô����
% 
%     % The size of the new matrix:
%         NyxzNew = [length(y1),length(x1),length(z1)]+2;
%         %ΪʲôҪ+2��չ����ά�ȣ�
% 
%     % The new indices in which the ROI will be located: 
%         x2 = x1 + (-x1(1) + 1 + round((NyxzNew(2) - length(x1))/2));
%         y2 = y1 + (-y1(1) + 1 + round((NyxzNew(1) - length(y1))/2)); 
%         z2 = z1 + (-z1(1) + 1 + round((NyxzNew(3) - length(z1))/2));
% %         t1 = x1(1)
% %         t2 = y1(1)
% %         t3=z1(1)
%         %������ֵƥ�䵽2��ͷ
%     % Embedding the new matrix:
%         image_3DNew = zeros(NyxzNew); 
%         image_3DNew(y2,x2,z2) = image_3D(y1, x1, z1); 
%         image_3D = image_3DNew; clear image_3DNew;
%     
% 	% The '3d-ish' colorbar: 
%     edge = 15;   
% %     k1=image_3D
%     image_3D = padarray(image_3D, [edge, edge, 0], 'post');
%     %����ͼ����� û�� padval����ʹ��0������䣬���и����15�У���ÿһά
%     %������15��
    
    N = size(image_3D);
    %��ȡ3D����ĳߴ�
    M = max(image_3D(:));
    %�ҵ����ֵM
%     b=image_3D(:);
%     for k = 1:edge
%         ny = N(1) - edge + k;
%         nx = N(2) - edge + k;
%         %�ҵ����ά0��Ԫ�ص�����
% %         a=1/0;
%         image_3D(ny,1:(k+1), round(k/(edge+1)*N(3)))  = 1/0;
%         image_3D(1:(k+1),nx, round(k/(edge+1)*N(3)))  = 1/0;
%         b=round(k/(edge+1)*N(3));
%         if k < edge
%             image_3D(ny,[k+1 ((k+ N(2)-edge))], :)  = 0;   
%             image_3D([k+1 ((k+ N(1)-edge))],nx,:)  = 0;   
%             image_3D(ny,(k+2):(k+ N(2)-(edge+1)) , round((k-0.5)/(edge)*N(3)))  = 2/3*M;
%             image_3D((k+2):(k+ N(1)-(edge+1)) ,nx, round((k-0.5)/(edge)*N(3))) = 1/3*M;
%             c=round((k-0.5)/(edge)*N(3));
%         else
%              image_3D(ny,(k+1):(k+ N(2)-edge) , :)  = 0;
%              image_3D((k+1):(k+N(1)-edge), nx, :)  = 0;
%         end
%     end
%         
    
    % color coding along the z-axis (using complex numbers)
    [X,Y,Z] = meshgrid(1:N(2), 1:N(1), (-N(3)/2:N(3)/2-1)/N(3));
%     ����һ����ά����ϵ��Z�����ϱ�ѹ������-9��8��18����ͬʱ����18���б��룬
%      ����Ϊ1/18
    Z = exp(-1i*(Z*1.75*pi));
    image_2D = (max(image_3D.*Z, [], 3)); 
%    Zÿһ��ƽ�涼һ�����൱��һ��Ȩ��

    % visual:
%     figureFULL;
   maxtric_W=vcc(image_2D); 
%     figureFULL;
%     subplot(1,2,1); vbw(abs(image_2D)); 
%     subplot(1,2,2); vcc(image_2D); 
end
    
function [vcc_rgb] = vcc(complexData2D, varargin)

%      figureFULL;
    
    %% Aim: visualize 2D complex image in rgb
    %����RGB��ʾ��λ����ͼ��
    
    % input: 
    % - complexData2D:  just what it says :-)
    % - string (optional) that determines the scaling: 
    %   no argument: linear intensity scaling
    %   's'        : root scaling
    %   'l'        : log scaling
    
    % magnitude displays the intensity
    % - black:        minimum intensity 
    % - full color:   maximum intensity
    
    % color diplays the phase 
    % - blue:   -pi
    % - green:  -pi/2
    % - yellow = 0;
    % - orange:  pi/2
    % - purple:  pi
    
    % One exception
    % - white: inf
 
    % jgbouwman@hotmail.com, 18 july 2014
    
    
    %% Scaling (default is linear)
    if isempty(varargin)
        scaling = 'linear';
    else
        scaling = varargin{1};
    end
    
    
    %% WHITE: 
    % All values that are infinite:
    whiteMask = double(uint8(isinf(complexData2D)));
    % ... initially set to zero:
    complexData2D(whiteMask==1) = 0;
    %ȥ��������ֵ
    %% COLOR: 
    % settings for the color coding: 
    rgbPEAK  = [6/16, -2/16, -0.55]*pi; % at which phase each channel peaksÿ���ŵ����ĸ���λ���ַ�ֵ
    rgbWIDTH = [1.4, 1, 0.4]; % a measure for the (relative) width of each peak ÿ����ģ���ԣ���ȵĶ��� 
    rgbMEAN  = [0.1, 0.1, 0.1]; % (MAX + MIN)/2 for each channel

    phase = angle(complexData2D);
    phaseColorMap = ones([size(complexData2D), 3]);
    %��ʼ��һ����ͨ����λͼ
    rgbMAXminMIN = 2*(1 - rgbMEAN);
    rgbMINIMUM   = 2*(rgbMEAN - 0.5);
    for c = 1:3
        if rgbWIDTH(c) > 1
            pow  = rgbWIDTH(c);
            peak = pi + rgbPEAK(c);
            phaseColorMap(:,:,c) =  1 -rgbMAXminMIN(c)*(cos(angle(exp(1i*(phase-peak))))/2+0.5).^pow;
        else
            pow  = 1/rgbWIDTH(c);
            peak = rgbPEAK(c);
            phaseColorMap(:,:,c) =  rgbMINIMUM(c) + rgbMAXminMIN(c)*(cos(angle(exp(1i*(phase-peak))))/2+0.5).^pow;
        end
    end
      
    % INCLUDE WHITE IN COLORMAP:
    phaseColorMap = max(phaseColorMap, repmat(whiteMask, [1,1,3]));

    
    %% SET INTENSITY (determined by magnitude):
    switch scaling
        case 'linear'
            magnIntensityMap = abs(complexData2D);
            magnIntensityMap = magnIntensityMap./max(magnIntensityMap(:));
        case 's'         
            magnIntensityMap = sqrt(abs(complexData2D));
            magnIntensityMap = magnIntensityMap./max(magnIntensityMap(:));
%              magnIntensityMap = (magnIntensityMap - min(magnIntensityMap(:)))/(max(magnIntensityMap(:)) - min(magnIntensityMap(:)));
        case 'l'
%             A = abs(complexData2D);
%             minA = min(min(A(A>0)))
%             maxA = max(max(A(A>0)))
%             A(A==0) = minA.^2/maxA;
%             minA = min(min(A(A>0)))
%             
            magnIntensityMap = log(eps + abs(complexData2D));
%             scale to [0-1]
            magnIntensityMap = (magnIntensityMap - min(magnIntensityMap(:)))/(max(magnIntensityMap(:)) - min(magnIntensityMap(:)));
    end
    

    
    % INCLUDE WHITE IN INTENSITY MAP:
    magnIntensityMap = repmat(max(magnIntensityMap, whiteMask), [1,1,3]);
    
    
    %% SHOW IT:
    vcc_rgb = phaseColorMap.*magnIntensityMap;
%     imagesc(vcc_rgb);
%     
%     axis equal tight;
%     axis off
%     set(gca,'xcolor','w','ycolor','w','xtick',[],'ytick',[])
%     set(gcf,'color','w')
%     set(gca,'box','on')
% %     set(gca,'box','off')
%     set(gca,'visible','off')
end

function [] = vbw(scalarData2D, varargin)

%  
    % jgbouwman@hotmail.com, 18 july 2014
    
    
    %% Scaling (default is linear)
    if isempty(varargin)
        scaling = 'linear';
        scalarData2D = scalarData2D;
    else
        scaling = varargin{1};
        
        if scaling == 's';
            scalarData2D = sqrt(scalarData2D);
        else
            scalarData2D = log(eps + abs(scalarData2D));
        end
    end
    
    %% SHOW IT:
    imagesc(scalarData2D); colormap gray; 
    
    axis equal tight;
    axis off
    set(gca,'xcolor','w','ycolor','w','xtick',[],'ytick',[])
    set(gcf,'color','w')
    set(gca,'box','on')
%     set(gca,'box','off')
    set(gca,'visible','off')
end

function figureFULL(varargin)
    if isempty(varargin)
        scrsz = get(0,'ScreenSize'); % full screen looks better
    else
        scrsz = varargin{1};
    end    
    figure('Position', [400,100,600,900]);
%     axes('Position',[0 0 1 1], 'Units','normalized');
end