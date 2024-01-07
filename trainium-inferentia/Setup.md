# Deep Learning On Inf2 Using the  Deep Learning AMI

To find the DLAMI execute the below command. Change the env `AWS_REGION` accordingly where Inferentia2 instances are available.

```
export AWS_REGION=us-west-2
DLAMIINF2=$(aws ec2 describe-images --region $AWS_REGION --owners amazon \
--filters 'Name=name,Values=Deep Learning AMI (Ubuntu 20.04) Version ??.?' 'Name=state,Values=available' \
--query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text)
```

Follow the prompts on the console and use the value of the DLAMIINF2 to launch an Inf2 EC2 of your choice. Create a kyepair file and download it locally.

Once the EC2 instance is in `Running` state, login to the instance

```
ssh -L localhost:8888:localhost:8888 -i "<Key-pair-file>.pem" ubuntu@<DNS Name of the instance>
```

Once logged in, install jupyter if not already installed using snap.

```
sudo snap install jupyter
jupyter notebook
```
Copy the URL that says `Copy/paste this URL into your browser`.
Now you can access the Jupyter Notebook Server from your local browser.
