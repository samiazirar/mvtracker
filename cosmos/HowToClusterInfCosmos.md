<!-- nvidia/cuda:12.2.2-runtime-ubuntu22.04 -->


enroot import -o /data/cosmos_predict2.sqsh docker://registry.gitlab.uni-bonn.de:5050/rpl/public_registry/cosmos-predict2:latest

export NVIDIA_DRIVER_CAPABILITIES=all
enroot create --force --name cosmos_inference /data/cosmos_predict2.sqsh
enroot start -r -w cosmos_inference


# pull the correction version

git pull

# if you modified
exit
enroot export --output ~/cosmos_inference_final.sqsh cosmos_inference



# finally run with


enroot create --force --name cosmos_inference_final ~/cosmos_inference_final.sqsh
enroot start -r -w cosmos_inference_final