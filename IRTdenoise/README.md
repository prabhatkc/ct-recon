## Total Variation (TV)-based Denoising (Post-processing) of low dose (noisy) CT images

```
usage: main.py [-h] --input-folder INPUT_FOLDER
               [--input-gen-folder INPUT_GEN_FOLDER]
               [--target-gen-folder TARGET_GEN_FOLDER]
               [--output-folder OUTPUT_FOLDER] [--loss-func LOSS_FUNC]
               [--prior-type PRIOR_TYPE] [--input-img-type INPUT_IMG_TYPE]
               [--lr LR] [--nite NITE] [--reg-lambda REG_LAMBDA] [--cuda]
               [--print-opt-errs] [--save-imgs] [--out-dtype OUT_DTYPE]

Application of total variation(TV)-based iterative denoisers on CT images

required & optional arguments:
  -h, --help            show this help message and exit
  --input-folder        directory name containing low dose images at different dose levels
  --input-gen-folder    directory name containing specific dose-level images for low dose
  --target-gen-folder   directory name containing Full dose imgs(if available). 
                        this option is not required for cases where Full dose is unavailable.
  --output-folder       main output foldername to store results
  --loss-func           loss function to be used such as mse, l1, ce
  --prior-type          prior terms to be combined with the data fedility
                        term. Options include l1, nl, sobel, tv-fd, tv-fbd
  --input-img-type      dicom or raw or tif?
  --lr 	                learning rate for a single GPU
  --nite 	              Number of iteration for each image
  --reg-lambda          pre-factor for the prior term (if used).
  --cuda                Use cuda?
  --print-opt-errs      print losses and error updates for each iteration?
  --save-imgs           save denoised images?
  --in-dtype            data type of input images. out image dtype is eq in-dtype.
```
### Example usage
`
python main.py --input-folder "./data/L096" --input-gen-folder "quarter_3mm_sharp_sorted" 
--cuda --input-img-type 'dicom' --lr 0.001 --nite 100 --reg-lambda 0.01 --save-imgs  --target-gen-folder "full_3mm_sharp_sorted"
--in-dtype 'uint16' --print-opt-errs`<br>
or<br>
$ chmod +x demo.sh<br>
$ ./demo

### References
- McCollough, C., Chen, B., Holmes III, D. R., Duan, X., Yu, Z., Yu, L., Leng, S., & Fletcher, J. (2020). Low Dose CT Image and Projection Data (LDCT-and-Projection-data) (Version 4) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/9NPB-2637

### Contact
Prabhat KC
prabhat.kc077@gmail.com
