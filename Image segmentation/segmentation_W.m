%%
close all;
clear all;

nombre_imagen = 'Blanco.jpg';
imagen = imread(nombre_imagen); 
imagen_original=rgb2gray(imagen);

ee=strel('disk',40);
openning=imopen(imagen_original,ee);
f1=imagen_original+openning;

ee=strel('disk',170);
cierre=imclose(f1,ee);
final=f1-imcomplement(cierre);

ee=strel('disk',20);
open=imopen(final,ee);
ee=strel('disk',20);
imagen2=imclose(open,ee);

figure, set(gcf, 'Name', 'Mejora de imagen', 'Position', get(0,'Screensize'))
subplot(2,2,1), imagesc(uint8(imagen_original)), axis off image, colormap gray, title('Imagen original')
subplot(2,2,2), imagesc(uint8(f1)), axis off image, colormap gray, title('Original + Apertura')
subplot(2,2,3), imagesc(uint8(final)), axis off image, colormap gray, title('Imagen - Cierre')
subplot(2,2,4), imagesc(uint8(imagen2)), axis off image, colormap gray, title('Imagen final')

filtro_LPF=1/25*ones(5);
filtrada=imfilter(imagen2,filtro_LPF);

%%
ee=strel('disk',220);
apertura=imclose(filtrada,ee);
segmentacion=apertura<175;

ee=strel('disk',240);
apertura_2=imclose(apertura,ee);
reconstruccion=imcomplement(imreconstruct(imcomplement(apertura_2),imcomplement(apertura)));
segmentacion_2=reconstruccion<175;

detectado=segmentacion-segmentacion_2;

figure, set(gcf, 'Name', 'Detecci?n de monedas', 'Position', get(0,'Screensize'))
subplot(2,3,1), imagesc(uint8(imagen)), axis off image, colormap gray, title('Imagen original')
subplot(2,3,2), imagesc(uint8(apertura)), axis off image, colormap gray, title('Primera apertura')
subplot(2,3,3), imagesc(uint8(apertura_2)), axis off image, colormap gray, title('Segunda apertura')
subplot(2,3,4), imagesc(uint8(segmentacion)), axis off image, colormap gray, title('Primera apertura umbralizada')
subplot(2,3,5), imagesc(uint8(segmentacion_2)), axis off image, colormap gray, title('Segunda apertura umbralizada')
subplot(2,3,6), imagesc(uint8(detectado)), axis off image, colormap gray, title('Moneda(s) detectada(s)')

detecta2=detectado.*double(imagen_original);
detecta22=imreconstruct(detecta2,double(imagen_original));
mascara=detecta22<125;
detecta2_2=mascara.*double(imagen);

figure, set(gcf, 'Name', 'Representaci?n de monedas', 'Position', get(0,'Screensize'))
subplot(1,3,1), imagesc(uint8(detectado)), axis off image, colormap gray, title('M?scara')
subplot(1,3,2), imagesc(uint8(detecta2)), axis off image, colormap gray, title('M?scara en la imagen BW')
subplot(1,3,3), imagesc(uint8(detecta2_2)), axis off image, colormap gray, title('M?scara reconstruida en la imagen RGB')


%% Blanco
% Hay que probar con varias im?genes
close all;
nombre_imagen = 'Blanco.jpg';
imagen = imread(nombre_imagen); 
imagen_original=rgb2gray(imagen);

ee=strel('disk',40);
openning=imopen(imagen_original,ee);
f1=imagen_original+openning;

ee=strel('disk',170);
cierre=imclose(f1,ee);
final=f1-imcomplement(cierre);

%
ee=strel('disk',20);
openning=imopen(final,ee);
ee=strel('disk',20);
imagen_defectos=imclose(openning,ee);
%

filtro_LPF=1/25*ones(5);
filtrada=imfilter(imagen_defectos,filtro_LPF);

figure, set(gcf, 'Name', 'Preprocesado de la imagen', 'Position', get(0,'Screensize'))
subplot(2,2,1), imagesc(uint8(imagen_original)), axis off image, colormap gray, title('Imagen original')
subplot(2,2,2), imagesc(uint8(f1)), axis off image, colormap gray, title('Imagen + Apertura')
subplot(2,2,3), imagesc(uint8(final)), axis off image, colormap gray, title('Imagen - Complemento (Cierre)')
subplot(2,2,4), imagesc(uint8(imagen_defectos)), axis off image, colormap gray, title('Imagen procesada')

radios=[220 240 262 275 288 295 315 325 380]
numeros=zeros(1,length(radios)-1);
monedas=[0.01 0.02 0.1 0.05 0.2 0.5 1 2];

detectado_final=0*double(filtrada);

for i=1:length(radios)-1
    ee=strel('disk',radios(i));
    apertura=imclose(filtrada,ee);
    segmentacion=apertura<175;

    ee=strel('disk',radios(i+1));
    apertura_2=imclose(apertura,ee);
    reconstruccion=imcomplement(imreconstruct(imcomplement(apertura_2),imcomplement(apertura)));
    segmentacion_2=reconstruccion<175;

    detectado=segmentacion-segmentacion_2;

    detecta2=detectado.*double(imagen_original);
    detecta22=imreconstruct(detecta2,double(imagen_original));
    mascara=detecta22<125;
    detecta2_2=mascara.*double(imagen);
    detectado_final=detecta2_2+detectado_final;
    
    figure, set(gcf, 'Name', 'Monedas', 'Position', get(0,'Screensize'))
    subplot(1,3,1), imagesc(uint8(detectado)), axis off image, colormap gray, title('M?scara detectada')
    subplot(1,3,2), imagesc(uint8(detecta2)), axis off image, colormap gray, title('Disco sobre imagen')
    subplot(1,3,3), imagesc(uint8(detecta2_2)), axis off image, colormap gray, title('M?scara reconstruida sobre imagen RGB')

    area=pi*radios(i+1)^2;
    numeros(i)=sum(sum(detectado))/area;
    disp(['Se ha(n) detectado ', num2str(round(numeros(i))),' moneda(s) de ',num2str(monedas(i)), '?'])
end
disp('')
disp(['?Enhorabuena! Tienes la tremenda cantidad de ', num2str(sum(monedas.*round(numeros))),'?'])

figure, set(gcf, 'Name', 'Monedas detectadas', 'Position', get(0,'Screensize'))
imagesc(uint8(detectado_final)), axis off image, colormap gray, title('M?scara detectada sobre monedas')
