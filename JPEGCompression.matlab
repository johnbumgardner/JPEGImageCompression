% Lossy Image Compression Project
% Author: John Bumgardner
% ECE 421 - Introduction to Signal Processing
clear all; close all;
% Load image
image = rgb2gray(imread("peppers.tiff"));
normalized_image = im2double(image); % Normalize the image
figure
imshow(normalized_image)
title('Normalized Image')
% Partition to patches
% Patches will be 8x8 pixels
P = 8*8; % number of pixels in a patch
% Just assume the image is square i guess
M = length(normalized_image);
N = length(normalized_image);
number_of_patches = ( M * N ) / P;
patches = 	[0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;];
for patch_num = 1:number_of_patches
	% Compute the x coordinate range
	if mod(patch_num, 64) == 0
		x_max = 512;
	else
		x_max = (mod(patch_num, 64))*8;
	end
	x_min = x_max - 7;
	% Compute the y coordinate range
	y_max = ceil(patch_num/64) * 8;
	y_min = y_max - 7;
	temp_y = y_min;
	temp_x = x_min;
	for x_index = 1:8
		for y_index = 1:8
			patches(x_index, y_index, patch_num) = normalized_image(temp_x, temp_y);
			temp_y = temp_y + 1;
		end
		temp_y = y_min;
		temp_x = temp_x + 1;
	end
end
%patches = reshape(image,8,8,[]);
%print out the first 3 patches of the image
figure
title('Patches of image');
subplot(1,3,1);
imshow(patches(:,:,1));
subplot(1,3,2);
imshow(patches(:,:,2));
subplot(1,3,3);
imshow(patches(:,:,3));
suptitle('Selected patches from image');
% Discrete Cosine Transform
dct_coeff = [0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;];
for index = 1:number_of_patches
	dct_coeff(:,:, index) = dct2(patches(:,:, index));
end
figure
subplot(1,3,1);
semilogy(sort(reshape(abs(dct_coeff(:,:, 1)), 64, 1)), '*');
subplot(1,3,2);
semilogy(sort(reshape(abs(dct_coeff(:,:, 2)), 64, 1)), '*');
subplot(1,3,3);
semilogy(sort(reshape(abs(dct_coeff(:,:, 3)), 64, 1)), '*');
suptitle('Selected sorted magnitudes');
avg_coeff = [0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;];
for index = 1:number_of_patches
	for i = 1:8
		for j = 1:8
			avg_coeff(i,j) = avg_coeff(i,j) + abs(dct_coeff(i,j,index));
		end
	end
end
for k = 1:8
	for l = 1:8
		avg_coeff(k,l) = avg_coeff(k,l) / number_of_patches;
	end
end
figure
surf(abs(log10(vec2mat(sort(reshape(abs(avg_coeff), 1, 64)),8))))
title('Average magnitudes of Coefficients');
% Quantization
%delta = [500, 300, 200, 100, 50, 20, 5]; %various deltas to quantize with
delta = [6, 5, 4, 3, 2, 1, .5];
quantized_coeff = [0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;];
approximate_patch = [0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;];
for image_iteration = 1:length(delta)
	for index = 1:number_of_patches
		for m = 1:8
			for n = 1:8
				quantized_coeff(m,n, index, image_iteration) =
				delta(image_iteration)*round(dct_coeff(m,n, index)/delta(image_iteration));
			end
		end
	end
end
for image_iteration = 1:length(delta)
	for index = 1:number_of_patches
		approximate_patch(:, :, index, image_iteration) = idct2(quantized_coeff(:,:, index,image_iteration));
	end
end
% time to stitch the image back together
images(512,512) = 0;
x_min = 0;
x_max = 0;
y_min = 0;
y_max = 0;
for image_index = 1:length(delta)
	for patch_num = 1:number_of_patches
		% Compute the x coordinate range
		if mod(patch_num, 64) == 0
			x_max = 512;
		else
			x_max = (mod(patch_num, 64))*8;
		end
		x_min = x_max - 7;
		% Compute the y coordinate range
		y_max = ceil(patch_num/64) * 8;
		y_min = y_max - 7;
		temp_y = y_min;
		temp_x = x_min;
		for x_index = 1:8
			for y_index = 1:8
				images(temp_x, temp_y ,image_index) = approximate_patch(x_index, y_index, patch_num, image_index);
				temp_y = temp_y + 1;
			end
			temp_y = y_min;
			temp_x = temp_x + 1;
		end
	end
end
figure
suptitle('Reconstructed Images');
errors(7) = 0;
for image_index = 1:length(delta)
	images(:,:,image_index) = im2double(images(:,:,image_index));
	subplot(2,4,image_index)
	error = immse(images(:,:, image_index), normalized_image);
	errors(image_index) = error;
	imshow(images(:,:, image_index));
	title("MSE = " + num2str(error) + " Delta = " + num2str(delta(image_index)))
end
figure
plot(flip(delta), errors)
xlabel('Quantization Levels')
ylabel('Mean Square Error')
title('Relationship between quantization and MSE')
%find percentages of nonzero coeff
fprintf("Quantization\t\t\tNon-zero percent\t\t\tDistortion\n")
nonzero_percent(length(delta)) = 0;
for i = 1:length(delta)
	nonzero_percent(i) = 100 * nnz(images(:,:,i)) / (N * M);
	fprintf("%d\t\t\t\t%d\t\t\t\t\t\t%d\t\t\t\t\n", delta(i), nonzero_percent(i), errors(i));
end
% Arithmetic Coding
% Get the planes - function of i
plane(number_of_patches, P) = 0;
plane_x_index = 0;
plane_y_index = 0;
for image_index = 1:length(delta)
	for i = 1:64
		for patch_index = 1:number_of_patches
			if mod(i, 8)==0
				plane_x_index = 8;
			else
				plane_x_index = mod(i, 8);
			end
		plane_y_index = ceil(i/8);
		plane(patch_index, i, image_index) =
		delta(image_index)*quantized_coeff(plane_x_index, plane_y_index , patch_index , image_index);
		end
	end
end
%get the counts of each plane
coding_rate(length(delta)) = 0
for image_index = 1:length(delta)
	%for i = 1:64
		i = 1;
		plane_modified = plane(:,i, image_index) + 1 - min(plane(:,i, image_index));
		counts = hist(plane_modified, 1:max(plane_modified)) + 1;
		seq = plane_modified;
		code_i = arithenco(abs(uint64(plane_modified)), counts); % encoder
		coding_rate(image_index) = (length(code_i) / M*N)
		decode_i = arithdeco(code_i, counts, length(plane_modified)); % decoder
		isequal(plane_modified, decode_i) % outputs 0 or 1
	%end
end
figure
plot(errors, log10(coding_rate))
xlabel('Error')
ylabel('Coding Rate')
title('Relationship of error and coding rate')
clear all; close all;
% Load image
image = rgb2gray(imread("flash.bmp"));
normalized_image = im2double(image); % Normalize the image
figure
imshow(normalized_image)
title('Normalized Image')
% Partition to patches
% Patches will be 8x8 pixels
P = 8*8; % number of pixels in a patch
% Just assume the image is square i guess
M = length(normalized_image);
N = length(normalized_image);
number_of_patches = ( M * N ) / P;
patches = [0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;];
for patch_num = 1:number_of_patches
	% Compute the x coordinate range
	if mod(patch_num, 64) == 0
		x_max = 512;
	else
		x_max = (mod(patch_num, 64))*8;
	end
	x_min = x_max - 7;
	% Compute the y coordinate range
	y_max = ceil(patch_num/64) * 8;
	y_min = y_max - 7;
	temp_y = y_min;
	temp_x = x_min;
	for x_index = 1:8
		for y_index = 1:8
			patches(x_index, y_index, patch_num) = normalized_image(temp_x, temp_y);
			temp_y = temp_y + 1;
		end
		temp_y = y_min;
		temp_x = temp_x + 1;
	end
end
%patches = reshape(image,8,8,[]);
%print out the first 3 patches of the image
figure
title('Patches of image');
subplot(1,3,1);
imshow(patches(:,:,400));
subplot(1,3,2);
imshow(patches(:,:,500));
subplot(1,3,3);
imshow(patches(:,:,600));
suptitle('Selected patches from image');
% Discrete Cosine Transform
dct_coeff = [0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;];
for index = 1:number_of_patches
	dct_coeff(:,:, index) = dct2(patches(:,:, index));
end
figure
subplot(1,3,1);
semilogy(sort(reshape(abs(dct_coeff(:,:, 400)), 64, 1)), '*');
subplot(1,3,2);
semilogy(sort(reshape(abs(dct_coeff(:,:, 800)), 64, 1)), '*');
subplot(1,3,3);
semilogy(sort(reshape(abs(dct_coeff(:,:, 600)), 64, 1)), '*');
suptitle('Selected sorted magnitudes');
avg_coeff = [0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;
			0, 0, 0 ,0, 0, 0, 0, 0;];
for index = 1:number_of_patches
	for i = 1:8
		for j = 1:8
			avg_coeff(i,j) = avg_coeff(i,j) + abs(dct_coeff(i,j,index));
		end
	end
end
for k = 1:8
	for l = 1:8
		avg_coeff(k,l) = avg_coeff(k,l) / number_of_patches;
	end
end
figure
surf(abs(log10(vec2mat(sort(reshape(abs(avg_coeff), 1, 64)),8))))
title('Average magnitudes of Coefficients');
% Quantization
%delta = [500, 300, 200, 100, 50, 20, 5]; %various deltas to quantize with
delta = [6, 5, 4, 3, 2, 1, .5];
quantized_coeff = [0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;
				0, 0, 0 ,0, 0, 0, 0, 0;];
approximate_patch = [0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;
					0, 0, 0 ,0, 0, 0, 0, 0;];
for image_iteration = 1:length(delta)
	for index = 1:number_of_patches
		for m = 1:8
			for n = 1:8
				quantized_coeff(m,n, index, image_iteration) =
				delta(image_iteration)*round(dct_coeff(m,n, index)/delta(image_iteration));
			end
		end
	end
end
for image_iteration = 1:length(delta)
	for index = 1:number_of_patches
		approximate_patch(:, :, index, image_iteration) = idct2(quantized_coeff(:,:, index, image_iteration));
	end
end
% time to stitch the image back together
images(512,512) = 0;
x_min = 0;
x_max = 0;
y_min = 0;
y_max = 0;
for image_index = 1:length(delta)
	for patch_num = 1:number_of_patches
		% Compute the x coordinate range
			if mod(patch_num, 64) == 0
				x_max = 512;
			else
				x_max = (mod(patch_num, 64))*8;
			end
			x_min = x_max - 7;
			% Compute the y coordinate range
			y_max = ceil(patch_num/64) * 8;
			y_min = y_max - 7;
			temp_y = y_min;
			temp_x = x_min;
			for x_index = 1:8
				for y_index = 1:8
					images(temp_x, temp_y ,image_index) = approximate_patch(x_index, y_index, patch_num, image_index);
					temp_y = temp_y + 1;
				end
				temp_y = y_min;
				temp_x = temp_x + 1;
			end
	end
end
figure
suptitle('Reconstructed Images');
errors(7) = 0;
for image_index = 1:length(delta)
	images(:,:,image_index) = im2double(images(:,:,image_index));
	subplot(2,4,image_index)
	error = immse(images(:,:, image_index), normalized_image);
	errors(image_index) = error;
	imshow(images(:,:, image_index));
	title("MSE = " + num2str(error) + " Delta = " + num2str(delta(image_index)))
end
figure
plot(flip(delta), errors)
xlabel('Quantization Levels')
ylabel('Mean Square Error')
title('Relationship between quantization and MSE')
%find percentages of nonzero coeff
fprintf("Quantization\t\t\tNon-zero percent\t\t\tDistortion\n")
nonzero_percent(length(delta)) = 0;
for i = 1:length(delta)
	nonzero_percent(i) = 100 * nnz(images(:,:,i)) / (N * M);
	fprintf("%d\t\t\t\t%d\t\t\t\t\t\t%d\t\t\t\t\n", delta(i), nonzero_percent(i), errors(i));
end
% Arithmetic Coding
% Get the planes - function of i
plane(number_of_patches, P) = 0;
plane_x_index = 0;
plane_y_index = 0;
for image_index = 1:length(delta)
	for i = 1:64
		for patch_index = 1:number_of_patches
			if mod(i, 8)==0
				plane_x_index = 8;
			else
				plane_x_index = mod(i, 8);
			end
			plane_y_index = ceil(i/8);
			plane(patch_index, i, image_index) = delta(image_index)*quantized_coeff(plane_x_index, plane_y_index , patch_index ,image_index);
		end
	end
end
error_flag = 0;
%get the counts of each plane
coding_rate(length(delta)) = 0;
for image_index = 1:6
	%for i = 1:64
		i = 1;
		plane_modified = plane(:,i, image_index) + 1 - min(plane(:,i, image_index));
		counts = hist(plane_modified, 1:max(plane_modified)) + 1;
		seq = plane_modified;
		code_i = arithenco(abs(uint64(plane_modified)), counts); % encoder
		coding_rate(image_index) = (length(code_i) / M*N);
		decode_i = arithdeco(code_i, counts, length(plane_modified)); % decoder
		if isequal(plane_modified, decode_i) == 0 % outputs 0 or 1
			error_flag = 1;
		end
	%end
end
if error_flag == 0 disp("No error in coding")
else disp("Error in coding")
end
figure
plot(errors, log10(coding_rate))
xlabel('Error')
ylabel('Coding Rate')
title('Relationship of error and coding rate')