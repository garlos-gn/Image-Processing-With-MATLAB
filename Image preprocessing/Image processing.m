%% Trabajo 1 PSAI

%% Recorte para anonimización
close all;

anonimizado = 'anonimizado.jpg'; 
imagen1 = single(imread(anonimizado));
imagen1 = mat2gray(imagen1,[0 255]);
t_im1=size(imagen1);
figure(1), set(gcf, 'Name', 'Recorte para anonimización', 'Position', get(0,'Screensize'))
subplot(1,3,1), imshow(imagen1), axis off image, title(['Imagen original (Tamaño = ' num2str(t_im1(1),3) 'x' num2str(t_im1(2),3) ')'])

imagen2=imcrop(imagen1,[20,71,600,603]); % Al no poner más argumentos en imcrop, el recorte se realizará manualmente
% imagen2=imcrop(imagen1);
t_im2=size(imagen2);
subplot(1,3,2), imshow(imagen2), axis off image, title(['Imagen anonimizada (Tamaño = ' num2str(t_im2(1),3) 'x' num2str(t_im2(2),3) ')'])

corregida = imadjust(imagen2,[0 0.8],[0 1]);
corregida2=rgb2hsv(corregida);
corregida2(:,:,1)=1*corregida2(:,:,1);
corregida2(:,:,2)=1.3*corregida2(:,:,2);
corregida2(:,:,3)=1*corregida2(:,:,3);
corregida=hsv2rgb(corregida2);
subplot(1,3,3), imshow(corregida), axis off image, title(['Imagen corregida (Tamaño = ' num2str(t_im2(1),3) 'x' num2str(t_im2(2),3) ')'])


%% Mejora de visualización
close all;

Correcciones = 'contraste.jpg';
Imagen_Correcciones = single(imread(Correcciones));
Imagen_Correcciones1 = mat2gray(Imagen_Correcciones,[0 255]);

Imagen_Correcciones = imadjust(Imagen_Correcciones1,[0.1 0.9],[0 1]);
rojo=Imagen_Correcciones(:,:,1);
verde=Imagen_Correcciones(:,:,2);
azul=Imagen_Correcciones(:,:,3);
[c1,h1] = imhist(rojo,255);
[c2,h2] = imhist(azul,255);
[c3,h3] = imhist(verde,255);

figure(1),set(gcf, 'Name', 'Mejora de visualización', 'Position', get(0,'Screensize'))
subplot(2,2,1), imshow(Imagen_Correcciones1,[0 1]);
axis off square, colormap gray, title('Image original')

subplot(2,2,2), imshow(Imagen_Correcciones,[0 1]);
axis off square, colormap gray, title('Imagen corregida con todos los canales')

rojo=imadjust(rojo,[0 0.6],[0 1]);
azul=imadjust(azul,[0 0.7],[0 1]);
verde=imadjust(verde,[0 0.6],[0 1]);

Imagen_Correcciones(:,:,1)=rojo;
Imagen_Correcciones(:,:,2)=verde;
Imagen_Correcciones(:,:,3)=azul;

subplot(2,2,3), imshow(Imagen_Correcciones,[0 1]);
axis off square, colormap gray, title('Imagen corregida canal a canal')

Imagen_Correcciones=imadjust(Imagen_Correcciones,[0 1],[0.01 0.95]);

Imagen_HSV=rgb2hsv(Imagen_Correcciones);
Imagen_HSV(:,:,1)=1.1*Imagen_HSV(:,:,1);
Imagen_HSV(:,:,2)=1.1*Imagen_HSV(:,:,2);
Imagen_HSV(:,:,3)=0.85*Imagen_HSV(:,:,3);
Imagen_Correcciones=hsv2rgb(Imagen_HSV);

subplot(2,2,4), imshow(Imagen_Correcciones,[0 1]);
axis off square, colormap gray, title('Imagen corregida canal a canal')

figure(2),set(gcf, 'Name', 'Mejora de visualización final', 'Position', get(0,'Screensize'))
subplot(1,3,1), imshow(Imagen_Correcciones1,[0 1]);
axis off square, colormap gray, title('Imagen original')

subplot(1,3,2), imshow(Imagen_Correcciones,[0 1]);
axis off square, colormap gray, title('Imagen corregida (imadjust)')

Imagen_Correcciones(:,:,1)=adapthisteq(Imagen_Correcciones(:,:,1),'ClipLimit',0.001);
Imagen_Correcciones(:,:,2)=adapthisteq(Imagen_Correcciones(:,:,2),'ClipLimit',0.001);
Imagen_Correcciones(:,:,3)=adapthisteq(Imagen_Correcciones(:,:,3),'ClipLimit',0.002);

subplot(1,3,3), imshow(Imagen_Correcciones,[0 1]);
axis off square, colormap gray, title('Imagen corregida (adapthisteq)')



%% Suavizado
close all;

Suavizado = 'suavizado.jpg';
Imagen_Suavizado = single(imread(Suavizado));
Imagen_Suavizado = mat2gray(Imagen_Suavizado,[0 255]);

figure(1),set(gcf, 'Name', 'Suavizado de imagen', 'Position', get(0,'Screensize'))
subplot(2,2,1), imshow(Imagen_Suavizado,[0 1]);
axis off square, colormap gray, title('Imagen original')

% Filtrado de mediana:
Ntimes=50;
N=4;
FiltroMediana1=Imagen_Suavizado(:,:,1);
FiltroMediana2=Imagen_Suavizado(:,:,2);
FiltroMediana3=Imagen_Suavizado(:,:,3);
resultado=Imagen_Suavizado;

figure(2), set(gcf, 'Name', 'Filtrado de mediana (espacio)', 'Position', get(0,'Screensize'))
figure(3), set(gcf, 'Name', 'Filtrado de mediana (frecuencia)', 'Position', get(0,'Screensize'))
for n=1:Ntimes
    FiltroMediana1 = medfilt2(FiltroMediana1, [1 1]*N);
    FiltroMediana2 = medfilt2(FiltroMediana2, [1 1]*N);
    FiltroMediana3 = medfilt2(FiltroMediana3, [1 1]*N);
    resultado(:,:,1)=FiltroMediana1;
    resultado(:,:,2)=FiltroMediana2;
    resultado(:,:,3)=FiltroMediana3;
    if(mod(n,5)==0)
        figure(2)
        subplot(2,5,n/5), imshow(resultado), axis off square, colormap gray, title(['Filtrado de mediana (Ntimes = ' num2str(n) ')']);
        figure(3)
        subplot(2,5,n/5), imagesc(1+log(abs(fftshift(fft2(rgb2gray(resultado)))))), axis off square, colormap gray, title(['Filtrado de mediana (Ntimes = ' num2str(n) ')']);
    end
end

figure(1)
subplot(2,2,2), imshow(resultado,[0 1]), axis off image, colormap copper, title('Corrección mediante filtro de Mediana');

% Filtrado de wiener
figure(4),set(gcf, 'Name', 'Filtrado de Wiener', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_Suavizado,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2),imagesc(1+log(abs(fftshift(fft2(rgb2gray(Imagen_Suavizado)))))), axis off image, colormap copper, title('Espectro de la imagen original')

Imagen_corregida=Imagen_Suavizado;
Imagen_corregida_1=wiener2(Imagen_Suavizado(:,:,1), [3 3]);
Imagen_corregida_2=wiener2(Imagen_Suavizado(:,:,2), [3 3]);
Imagen_corregida_3=wiener2(Imagen_Suavizado(:,:,3), [3 3]);
Imagen_corregida(:,:,1)=Imagen_corregida_1;
Imagen_corregida(:,:,2)=Imagen_corregida_2;
Imagen_corregida(:,:,3)=Imagen_corregida_3;

subplot(2,2,3),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida mediante filtrado de Wiener')
TF2=fft2(rgb2gray(Imagen_corregida));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Espectro de la imagen corregida mediante filtrado de Wiener')

figure,set(gcf, 'Name', 'Comparación filtrado de wiener', 'Position', get(0,'Screensize'))
subplot(1,2,1),imshow(Imagen_Suavizado,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(1,2,2),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtrado de Wiener)')

figure(1)
subplot(2,2,3), imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Corrección mediante filtrado de Wiener');

% Filtrado en frecuencia

for N=1:5
    TF=fft2(Imagen_Suavizado);
    D0 = 0.03*N; n = 3;
    [u,v] = freqspace([1123 1125],'meshgrid'); % Only for MATLAB
    D = sqrt(u.^2+v.^2);
    H2 = fftshift(1./(1+(D./D0).^(2*n)));
    figure, set(gcf, 'Name', 'Filtro paso bajo de Butterworth', 'Position', get(0,'Screensize'))
    subplot(2,2,1), imagesc(fftshift(H2)), axis off image, colormap copper, title(['Respuesta en frecuencia del filtro. D0 = ' num2str(D0)])
    subplot(2,2,2), imagesc(1+log(abs(fftshift(fft2(rgb2gray(Imagen_Suavizado)))))), axis off image, colormap copper, title('TF de la imagen original')
    TF_resultado_LPF=H2.*TF;
    resultado_LPF=real(ifft2(TF_resultado_LPF));
    subplot(2,2,3), imshow(resultado_LPF, [0 1]), axis off image, colormap copper, title('Imagen suavizada')
    subplot(2,2,4), imagesc(1+log(abs(fftshift(fft2(rgb2gray(resultado_LPF)))))), axis off image, colormap copper, title('Respuesta en frecuencia de la imagen suavizada')
end

figure(1)
subplot(2,2,4), imshow(resultado_LPF,[0 1]), axis off image, colormap copper, title('Filtro Paso Bajo de Butterworth')


% Parece coherente pensar que un filtrado de wiener tras un filtrado
% mediante mediana proporcionará unos resultados considerablemente buenos,
% por lo que será este el enfoque que tomaremos

Ntimes=20;
N=4;
FiltroMediana1=Imagen_Suavizado(:,:,1);
FiltroMediana2=Imagen_Suavizado(:,:,2);
FiltroMediana3=Imagen_Suavizado(:,:,3);
resultado=Imagen_Suavizado;

for n=1:Ntimes
    FiltroMediana1 = medfilt2(FiltroMediana1, [1 1]*N);
    FiltroMediana2 = medfilt2(FiltroMediana2, [1 1]*N);
    FiltroMediana3 = medfilt2(FiltroMediana3, [1 1]*N);
    resultado(:,:,1)=FiltroMediana1;
    resultado(:,:,2)=FiltroMediana2;
    resultado(:,:,3)=FiltroMediana3;
end


TF=fft2(resultado);
D0 = 0.2; n = 3;
[u,v] = freqspace([1123 1125],'meshgrid');
D = sqrt(u.^2+v.^2);
H2 = fftshift(1./(1+(D./D0).^(2*n)));
TF_resultado_LPF=H2.*TF;
Imagen_corregida=real(ifft2(TF_resultado_LPF));

figure, set(gcf, 'Name', 'Filtrado seleccionado', 'Position', get(0,'Screensize'))
subplot(2,2,1), imshow(Imagen_Suavizado,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,3), imagesc(1+log(abs(fftshift(fft2(rgb2gray(Imagen_Suavizado)))))), axis off image, colormap copper, title('Respuesta en frecuencia de la imagen suavizada')
subplot(2,2,4), imagesc(1+log(abs(fftshift(fft2(rgb2gray(Imagen_corregida)))))), axis off image, colormap copper, title('Respuesta en frecuencia de la imagen suavizada')
subplot(2,2,2), imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida')


a=size(Imagen_corregida);
TF=fft2(Imagen_corregida);
TF_BN=fft2(rgb2gray(Imagen_corregida));

D0 = 0.08; n = 1;
[u,v] = freqspace([a(1) a(2)],'meshgrid');
D = sqrt(u.^2+v.^2);
H2 = fftshift(1./(1+(D0./D).^(2*n)));
figure, set(gcf, 'Name', 'HPF de  Butterworth', 'Position', get(0,'Screensize'))
subplot(2,2,1), imagesc(fftshift(H2)), axis off image, colormap copper, title(['Respuesta en frecuencia del filtro. D0 = ' num2str(D0)])
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF_BN)))), axis off image, colormap copper, title('Respuesta en frecuencia de la imagen original')
TF_resultado_HPF=H2.*TF;
resultado_HPF=real(ifft2(TF_resultado_HPF));
resultado_HPF1=rgb2gray(resultado_HPF);
subplot(2,2,3), imshow(resultado_HPF,[0 1]), axis off image, colormap copper, title('Imagen filtrada')
subplot(2,2,4), imagesc(1+log(abs(fftshift(fft2(resultado_HPF1))))), axis off image, colormap copper, title('Espectro de la imagen filtrada')

figure, set(gcf, 'Name', 'HPF de  Butterworth', 'Position', get(0,'Screensize'))
subplot(1,3,1), imshow(Imagen_Suavizado,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(1,3,2), imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen suavizada')
subplot(1,3,3), imshow(resultado_HPF+Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen con detalles añadidos')

%% Realzado
close all;

Realzado = 'realzado.tif';
Imagen_Realzado=single(imread(Realzado));
Imagen_Realzado=mat2gray(Imagen_Realzado,[0 255]);
figure(1),set(gcf, 'Name', 'Realzado', 'Position', get(0,'Screensize'))

% Filtrado en el espacio
for alpha=[0.05 0.2 0.5 0.8 0.95]
    f_laplaciano=fspecial('laplacian',alpha)
    Laplacian=-imfilter(Imagen_Realzado, f_laplaciano);
    figure,set(gcf, 'Name', 'Realzado', 'Position', get(0,'Screensize'))
    subplot(1,3,1), imshow(Imagen_Realzado,[0 1]), axis off image, colormap copper, title(['Imagen original (alpha = ' num2str(alpha) ')'])
    subplot(1,3,2), imshow(Laplacian,[0 1]), axis off image, colormap copper, title('Detalles de la imagen')
    subplot(1,3,3), imshow(Imagen_Realzado+Laplacian,[0 1]), axis off image, colormap copper, title('Imagen realzada (LSI)')  
end

figure(1)
subplot(1,3,1), imshow(Imagen_Realzado+Laplacian, [0 1]), axis off image, colormap copper, title('Imagen realzada (LSI)')

% Filtrado en la frecuencia
TF=fft2(Imagen_Realzado);
TF_BN=fft2(rgb2gray(Imagen_Realzado));
a=size(Imagen_Realzado);

D0 = 0.2; n = 3;
[u,v] = freqspace([a(1) a(2)],'meshgrid');
D = sqrt(u.^2+v.^2);
H2 = fftshift(1./(1+(D0./D).^(2*n)));
figure, set(gcf, 'Name', 'LPF de Butterworth', 'Position', get(0,'Screensize'))
subplot(2,2,1), imagesc(fftshift(H2)), axis off image, colormap copper, title(['Filter frecuency response (image representation). D0 = ' num2str(D0)])
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF_BN)))), axis off image, colormap copper, title('Respuesta en frecuencia de la imagen original')
TF_resultado_HPF=H2.*TF;
resultado_HPF=real(ifft2(TF_resultado_HPF));
resultado_HPF1=rgb2gray(resultado_HPF);
subplot(2,2,3), imshow(resultado_HPF,[0 1]), axis off image, colormap copper, title('Imagen filtrada')
subplot(2,2,4), imagesc(1+log(abs(fftshift(fft2(resultado_HPF1))))), axis off image, colormap copper, title('Espectro de la imagen filtrada')

figure(1)
subplot(1,3,2), imshow(Imagen_Realzado+resultado_HPF, [0 1]), axis off image, colormap copper, title('Imagen realzada en frecuencia')

Imagen_Realzado=imadjust(Imagen_Realzado+resultado_HPF,[0.1 1],[0 1]);
Imagen_HSV=rgb2hsv(Imagen_Realzado);
Imagen_HSV(:,:,1)=1.1*Imagen_HSV(:,:,1);
Imagen_HSV(:,:,2)=1.1*Imagen_HSV(:,:,2);
Imagen_HSV(:,:,3)=0.9*Imagen_HSV(:,:,3);
Imagen_Realzado=hsv2rgb(Imagen_HSV);
subplot(1,3,3), imshow(Imagen_Realzado, [0 1]), axis off image, colormap copper, title('Imagen tras corrección de color')


%% Fusión de información
close all;
clc

multi1 = 'multiespectral1.tiff';
multi2 = 'multiespectral2.tiff';
multi3 = 'multiespectral3.tiff';
M1 = single(imread(multi1));
M2 = single(imread(multi2));
M3 = single(imread(multi3));
M1 = mat2gray(M1,[0 255]);
M2 = mat2gray(M2,[0 255]);
M3 = mat2gray(M3,[0 255]);

figure(1),set(gcf, 'Name', 'Imagen multiespectral', 'Position', get(0,'Screensize'))
subplot(1,3,1), imshow(M1), axis square off, title('Primera imagen')
subplot(1,3,2), imshow(M2), axis square off, title('Segunda imagen')
subplot(1,3,3), imshow(M3), axis square off, title('Tercera imagen')

verde=M1(:,:,2);
azul=M2(:,:,3);
rojo=M3(:,:,1);

[c1,h1] = imhist(rojo,255);
[c2,h2] = imhist(azul,255);
[c3,h3] = imhist(verde,255);

figure(2),set(gcf, 'Name', 'Contrast Adjustment', 'Position', get(0,'Screensize'))
subplot(1,3,1), bar(h1*255,c1/sum(c1),'stacked','r');
axis square off, axis([-2 255 0 0.04]), title('Histograma imagen original (rojo)'), colorbar('XTickLabel','','location','North')
subplot(1,3,2), bar(h2*255,c2/sum(c2),'stacked','g');
axis square off, axis([-2 255 0 0.04]), title('Histograma imagen original (verde)'), colorbar('XTickLabel','','location','North')
subplot(1,3,3), bar(h3*255,c3/sum(c3),'stacked','b');
axis square off, axis([-2 255 0 0.04]), title('Histograma imagen original (azul)'), colorbar('XTickLabel','','location','North')

imagen_combinada=M1;
imagen_combinada(:,:,1)=rojo;
imagen_combinada(:,:,2)=verde;
imagen_combinada(:,:,3)=azul;

figure(3),set(gcf, 'Name', 'Contrast Adjustment', 'Position', get(0,'Screensize'))
subplot(1,2,1),imshow(imagen_combinada), axis off image, colormap copper, title('Imagen combinada')

imagen_combinada_corregida=imagen_combinada;

Imagen_HSV=rgb2hsv(imagen_combinada_corregida);
Imagen_HSV(:,:,1)=1*Imagen_HSV(:,:,1);
Imagen_HSV(:,:,2)=1.05*Imagen_HSV(:,:,2);
Imagen_HSV(:,:,3)=0.65*Imagen_HSV(:,:,3);
imagen_combinada_corregida=hsv2rgb(Imagen_HSV);
imagen_combinada_corregida = imadjust(imagen_combinada_corregida,[0 0.75],[0 1]);
subplot(1,2,2),imshow(imagen_combinada_corregida), axis off image, colormap copper, title('Imagen combinada tras la mejora de color')

%% Caracterización y eliminación del ruido
close all;

Ruido = 'ruido.tif';
Imagen_Ruido = single(imread(Ruido));
Imagen_Ruido = mat2gray(Imagen_Ruido,[0 255]);

TF=fft2(Imagen_Ruido);
TF1=fft2(rgb2gray(Imagen_Ruido));

figure(1),set(gcf, 'Name', 'Comparación en el espacio', 'Position', get(0,'Screensize'))
subplot(2,3,1), imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')

figure(2),set(gcf, 'Name', 'Comparación en el espectro', 'Position', get(0,'Screensize'))
subplot(2,3,1), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('Imagen original')


figure,set(gcf, 'Name', 'Caracterización del ruido', 'Position', get(0,'Screensize'))
subplot(2,2,1), imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('TF de la imagen original')
v=round(ginput());
BWmask=roipoly(Imagen_Ruido,v(:,1),v(:,2));
[cr,hr]=imhist(Imagen_Ruido(BWmask));
subplot(2,2,3), bar(hr*255,cr/sum(cr),'stacked'),axis square off, title('Histograma del patrón dentro de la imagen'), colorbar('XTickLabel','','location','North')
subplot(2,2,4), imshow(Imagen_Ruido.*BWmask,[0 1]), axis off image, colormap copper, title('Patrón seleccionado sobre la imagen original')


%Vamos a usar un filtrado de wiener:
figure,set(gcf, 'Name', 'Filtrado de Wiener', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2),imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('Espectro de la imagen original')

Imagen_corregida=Imagen_Ruido;
Imagen_corregida_1=wiener2(Imagen_Ruido(:,:,1), [8 8]);
Imagen_corregida_2=wiener2(Imagen_Ruido(:,:,2), [8 8]);
Imagen_corregida_3=wiener2(Imagen_Ruido(:,:,3), [8 8]);
Imagen_corregida(:,:,1)=Imagen_corregida_1;
Imagen_corregida(:,:,2)=Imagen_corregida_2;
Imagen_corregida(:,:,3)=Imagen_corregida_3;
subplot(2,2,3),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de wiener)')
TF2=fft2(rgb2gray(Imagen_corregida));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Espectro de la imagen corregida (filtro de wiener)')

figure(1)
subplot(2,3,2), imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de wiener)')

figure(2),set(gcf, 'Name', 'Comparación en el espectro', 'Position', get(0,'Screensize'))
subplot(2,3,2), imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Imagen corregida (filtro de wiener)')

%Descubrimos que es un ruido gaussiano -> Usamos un filtro de media
figure,set(gcf, 'Name', 'Filtro de media aritmética', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('TF de la imagen original')

Imagen_corregida=imfilter(Imagen_Ruido,fspecial('average', [3 3]),'replicate');
subplot(2,2,3),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de media aritmética)')
TF2=fft2(rgb2gray(Imagen_corregida));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Espectro de la imagen corregida (filtro de media aritmética)')

figure(1)
subplot(2,3,3), imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de media aritmética)')

figure(2)
subplot(2,3,3), imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Imagen corregida (filtro de media aritmética)')


%Usamos filtro geométrico
figure,set(gcf, 'Name', 'Filtro geométrico', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('TF de la imagen original')
for i=1:3
    Imagen_corregida(:,:,i)=colfilt(Imagen_Ruido(:,:,i), [3 3],'sliding', @geomean);
end

subplot(2,2,3),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de media geométrica)')
TF3=fft2(rgb2gray(Imagen_corregida));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF3)))), axis off image, colormap copper, title('Imagen corregida (filtro de media geométrica)')

figure(1)
subplot(2,3,4), imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de media geométrica)')

figure(2)
subplot(2,3,4), imagesc(1+log(abs(fftshift(TF3)))), axis off image, colormap copper, title('Imagen corregida (filtro de media geométrica)')


%Vamos a usar un LPF
D0 = 0.2; n = 5;
[u,v] = freqspace([620 620],'meshgrid');
D = sqrt(u.^2+v.^2);
H2 = fftshift(1./(1+(D./D0).^(2*n)));
figure, set(gcf, 'Name', 'Butterworth lowpass filter', 'Position', get(0,'Screensize'))
subplot(2,2,1), imagesc(fftshift(H2)), axis off image, colormap copper, title(['Respuesta en frecuencia del filtro. D0 = ' num2str(D0)])
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('Espectro de la imagen original')
TF_resultado_LPF=H2.*TF;
imagen_corregida=real(ifft2(TF_resultado_LPF));
subplot(2,2,3), imshow(imagen_corregida), axis off image, colormap copper, title('Imagen filtrada (LPF de Butterworth)')
resultado_LPF=rgb2gray(imagen_corregida);
subplot(2,2,4), imagesc(1+log(abs(fftshift(fft2(resultado_LPF))))), axis off image, colormap copper, title('Espectro de la imagen filtrada')

figure(1)
subplot(2,3,5), imshow(imagen_corregida), axis off image, colormap copper, title('Imagen filtrada (LPF de Butterworth)')

figure(2)
subplot(2,3,5), imagesc(1+log(abs(fftshift(fft2(resultado_LPF))))), axis off image, colormap copper, title('Imagen corregida (LPF de Butterworth)')

%Vamos a usar un filtro olímpico
Imagen_corregida2=Imagen_Ruido;
figure,set(gcf, 'Name', 'Filtro olímpico', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('Espectro de la imagen original')
for i=1:3
    percent=20;
    Imagen_corregida2(:,:,i)=colfilt(Imagen_Ruido(:,:,i), [3 3], 'sliding', @(x) trimmean(x,percent));
end
subplot(2,2,3),imshow(Imagen_corregida2,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro olímpico)')
TF4=fft2(rgb2gray(Imagen_corregida2));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF4)))), axis off image, colormap copper, title('Espectro de la imagen corregida')

figure(1)
subplot(2,3,6), imshow(Imagen_corregida2,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro olímpico)')

figure(2)
subplot(2,3,6), imagesc(1+log(abs(fftshift(TF4)))), axis off image, colormap copper, title('Imagen corregida (filtro olímpico)')

%Enfoque seguido: filtro olímpico
imagen_corregida=Imagen_Ruido;
figure,set(gcf, 'Name', 'Filtro olímpico', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_Ruido,[0 1]), axis off image, colormap copper, title('Imagen original')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF1)))), axis off image, colormap copper, title('Espectro de la imagen original')
for i=1:3
    percent=40;
    imagen_corregida(:,:,i)=colfilt(Imagen_Ruido(:,:,i), [3 3], 'sliding', @(x) trimmean(x,percent));
end
subplot(2,2,3),imshow(imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro olímpico)')
TF4=fft2(rgb2gray(imagen_corregida));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF4)))), axis off image, colormap copper, title('Espectro de la imagen corregida')


%Usamos filtro geométrico
figure,set(gcf, 'Name', 'Filtro geométrico', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen tras el filtrado olímpico')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF4)))), axis off image, colormap copper, title('Espectro de la imagen tras el filtrado olímpico')
for i=1:3
    Imagen_corregida(:,:,i)=colfilt(abs(imagen_corregida(:,:,i)), [3 3],'sliding', @geomean);
end

subplot(2,2,3),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de media geométrica)')
TF3=fft2(rgb2gray(Imagen_corregida));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF3)))), axis off image, colormap copper, title('Espectro de la imagen corregida (filtro de media geométrica)')

figure,set(gcf, 'Name', 'Filtrado de Wiener', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_corregida,[0 1]), axis off image, colormap copper, title('Imagen tras filtrado geométrico')
subplot(2,2,2),imagesc(1+log(abs(fftshift(TF3)))), axis off image, colormap copper, title('Espectro de la imagen tras filtrado geométrico')

Imagen_corregida2=Imagen_corregida;
Imagen_corregida_1=wiener2(Imagen_corregida2(:,:,1), [3 3]);
Imagen_corregida_2=wiener2(Imagen_corregida2(:,:,2), [3 3]);
Imagen_corregida_3=wiener2(Imagen_corregida2(:,:,3), [3 3]);
Imagen_corregida2(:,:,1)=Imagen_corregida_1;
Imagen_corregida2(:,:,2)=Imagen_corregida_2;
Imagen_corregida2(:,:,3)=Imagen_corregida_3;
subplot(2,2,3),imshow(Imagen_corregida2,[0 1]), axis off image, colormap copper, title('Imagen corregida (filtro de wiener)')
TF2=fft2(rgb2gray(Imagen_corregida2));
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Espectro de la imagen corregida (filtro de wiener)')

Imagen_HSV=rgb2hsv(Imagen_corregida2);
Imagen_HSV(:,:,1)=1*Imagen_HSV(:,:,1);
Imagen_HSV(:,:,2)=1.05*Imagen_HSV(:,:,2);
Imagen_HSV(:,:,3)=0.8*Imagen_HSV(:,:,3);
Imagen_final=hsv2rgb(Imagen_HSV);
TF_fin=fft2(rgb2gray(Imagen_final));

figure,set(gcf, 'Name', 'Iluminación (HSV)', 'Position', get(0,'Screensize'))
subplot(2,2,1),imshow(Imagen_corregida2,[0 1]), axis off image, colormap copper, title('Imagen tras filtrado de Wiener')
subplot(2,2,3),imagesc(1+log(abs(fftshift(TF2)))), axis off image, colormap copper, title('Espectro de la imagen tras filtrado de Wiener')
subplot(2,2,2),imshow(Imagen_final,[0 1]), axis off image, colormap copper, title('Imagen corregida (con corrección de HSV)')
subplot(2,2,4),imagesc(1+log(abs(fftshift(TF_fin)))), axis off image, colormap copper, title('Espectro de la imagen corregida')

figure,set(gcf, 'Name', 'Caracterización del ruido de la imagen final', 'Position', get(0,'Screensize'))
subplot(2,2,1), imshow(Imagen_final,[0 1]), axis off image, colormap copper, title('Imagen corregida')
subplot(2,2,2), imagesc(1+log(abs(fftshift(TF_fin)))), axis off image, colormap copper, title('Espectro de la imagen original')
v=round(ginput());
BWmask=roipoly(Imagen_final,v(:,1),v(:,2));
[cr,hr]=imhist(Imagen_final(BWmask));
subplot(2,2,3), bar(hr*255,cr/sum(cr),'stacked'),axis square off, title('Histograma de la máscara seleccionada'), colorbar('XTickLabel','','location','North')
subplot(2,2,4), imshow(Imagen_final.*BWmask,[0 1]), axis off image, colormap copper, title('Patrón seleccionado sobre la imagen corregida')


%% Detección de patrones
clear all;
close all;

Patrones = 'patrones.png';
Imagen_Patrones = single(imread(Patrones));
Imagen_Patrones = mat2gray(Imagen_Patrones,[0 255]);
patrones_gray=rgb2gray(Imagen_Patrones);

ImageCovariance = @(A,B) conv2(A-mean(A(:)), B(end:-1:1,end:-1:1)-mean(B(:)),'valid');
ImageCorrelation = @(A,B) conv2(A, B(end:-1:1,end:-1:1),'valid');
ImageAutocovariance = @(A,B) conv2(A.*A,ones(size(B)),'valid')-conv2(A,ones(size(B)),'valid').^2/numel(B);
ImageQuadraticMean = @(A,B) conv2(A.*A,ones(size(B)),'valid');

CorrelationThreshold = 0.6;
figure(1), set(gcf, 'Name', 'Patrones', 'Position', get(0,'Screensize'))
imshow(Imagen_Patrones,[0 1]), axis off square, colormap gray, title('Imagen original')

indices=round(ginput(2));

Pattern=Imagen_Patrones(indices(1,2):indices(2,2), indices(1,1):indices(2,1),:);

figure(2), set(gcf, 'Name', 'Patrón a detectar en la imagen', 'Position', get(0,'Screensize'))
subplot(2,2,1), imshow(Pattern,[0 1]), axis off square, colormap gray, title('Patrón a identificar')
subplot(2,2,2), mesh(Pattern(:,:,1)), axis off square, set(gca,'XDir','reverse'), title('Elevación del patrón a identificar (canal rojo)')
subplot(2,2,3), mesh(Pattern(:,:,2)), axis off square, set(gca,'XDir','reverse'), title('Elevación del patrón a identificar (canal verde)')
subplot(2,2,4), mesh(Pattern(:,:,3)), axis off square, set(gca,'XDir','reverse'), title('Elevación del patrón a identificar (canal azul)')

for i=1:3
    PearsonCorrelationCoefficient = ImageCovariance(Imagen_Patrones(:,:,i),Pattern(:,:,i));
    PearsonCorrelationCoefficientDem = sqrt(ImageAutocovariance(Imagen_Patrones(:,:,i),Pattern(:,:,i)).*ImageAutocovariance(Pattern(:,:,i),Pattern(:,:,i)));
    Pearson(:,:,i)=PearsonCorrelationCoefficient;
    Pearson_dem(:,:,i)=PearsonCorrelationCoefficientDem;
end
index = find(Pearson_dem ~= 0);
Pearson(index) = Pearson(index)./Pearson_dem(index);

for i=1:3
    Pearson2(:,:,i) = padarray(Pearson(:,:,i),floor((size(Imagen_Patrones(:,:,1))-size(Pearson(:,:,1)))/2),0,'post');
    Pearson3(:,:,i) = padarray(Pearson2(:,:,i),ceil((size(Imagen_Patrones(:,:,1))-size(Pearson(:,:,1)))/2),0,'pre');
    Pearson4(:,:,i) = Pearson3(:,:,i).*(Pearson3(:,:,i)>CorrelationThreshold);
end

indice=find(Pearson4 ~= 0);
Pearson4(indices)=1;

figure, set(gcf, 'Name', 'Resultado de la localización de patrones', 'Position', get(0,'Screensize'))
subplot(1,3,1), imshow(Imagen_Patrones,[0 1]),axis off square, colormap gray, title('Imagen original') 
subplot(1,3,2), imshow(Imagen_Patrones+Pearson4,[0 1]),axis off square, colormap gray, title('Patrones identificados en la Imagen original') 
subplot(1,3,3), imshow(Pearson4), axis off square, set(gca,'YDir','reverse'), title('Patrones identificados')

