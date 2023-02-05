# Denoise (Post-process) low dose (noisy) CT images using bilateral filter.

```
usage: main.py [-h] --input-folder INPUT_FOLDER
               [--input-gen-folder INPUT_GEN_FOLDER]
               [--target-gen-folder TARGET_GEN_FOLDER]
               [--output-folder OUTPUT_FOLDER]
               [--input-img-type INPUT_IMG_TYPE] [--save-imgs]
               [--in-dtype IN_DTYPE] [--win-size WIN_SIZE]
               [--sigma-color SIGMA_COLOR] [--sigma-spatial SIGMA_SPATIAL]
               [--rNx RNX]

Application of bilateral filter-based denoisers on CT images

required & optional arguments:
  -h, --help            show this help message and exit
  --input-folder        directory name containing low dose images at different dose levels
  --input-gen-folder    directory name containing specific dose-level images for low dose
  --target-gen-folder   directory name containing Full dose imgs(if available). 
                        this option is not required for cases where Full dose is unavailable.
  --output-folder       main output foldername to store denoised results.
  --input-img-type      dicom or raw or tif?
  --save-imgs           save denoised images?
  --in-dtype            data type of input images. out-dtype is eq to in-
                        dtype.
  --win-size            window size for filtering
  --sigma-color         std for intensity. Larger value results in averaging
                        of larger intensity differences.
  --sigma-spatial       std for pixel distance. Larger value results in
                        averaging of pixels in larger distance.
  --rNx                 image size for raw image as input.
```
## Example usage ##
```
python main.py --input-folder '../data/phandata/poisson' --input-gen-folder 'noisy' --input-img-type 'raw' \
--save-imgs  --target-gen-folder 'gt' --in-dtype 'uint16' --sigma-color 0.02 \
--sigma-spatial 5 --win-size 7 --rNx 512 --output-folder './results/phandata'
```
> > Instead you may choose to (modify &) execute demo.sh file as
```
$ chmod +x demo.sh<br>
$ ./demo
```
## References ##
- This implementation imports bilteral filtering library from the [scikit-image's](https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise.html).

## Contact ##
Prabhat KC
prabhat.kc077@gmail.com
