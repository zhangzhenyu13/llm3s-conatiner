export CUDA_VISIBLE_DEVICES=1 

wbits=8
gsize=128
python quant_bloom.py \
    ~/CommonModels/your-org/bloomS2/ \
    xz-instruct \
    --wbits ${wbits} \
    --groupsize ${gsize}  \
    --save ~/CommonModels/your-org/bloomS2/gptq-${wbits}bit-${gsize}g.pt
echo
echo
echo "*****finished ${wbits}b-${gsize}g*****"
echo

