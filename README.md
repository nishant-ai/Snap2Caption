# Snap2Caption Terraform Setup

This repository contains the Terraform setup for provisioning instances on **KVM@TACC** and **CHI@UC** via OpenStack. The setup includes automatic Floating IP allocation, SSH access, and seamless integration with the `Snap2Caption` application.

---

## **Infrastructure Setup**
The Terraform configuration provisions the following resources:

1. **VM Instances:**
   - Two Virtual Machines:
     - `mlops-kvm-vm` on `KVM@TACC`
     - `mlops-chiuc-vm` on `CHI@UC`
   - Image: `CC-Ubuntu20.04`
   - Flavors:
     - `m1.small` for KVM
     - `baremetal` for CHI@UC
   - Key Pair: `id_rsa_chameleon`

2. **Floating IPs:**
   - Allocated dynamically using the `public-floating` pool.

3. **Security Groups:**
   - Default security group for SSH access.

---

## **Pre-requisites**
1. Terraform CLI (>= 1.0.0)  
2. OpenStack CLI  
3. AWS CLI (if using S3 for remote state storage)  
4. SSH keypair named `id_rsa_chameleon` configured in OpenStack  

---

## **Steps to Deploy**
1. **Navigate to the Terraform Directory:**
    ```bash
    cd kvm-terraform
    ```

2. **Initialize Terraform:**
    ```bash
    terraform init
    ```

3. **Verify the Plan:**
    ```bash
    terraform plan -var-file="terraform.tfvars"
    ```

4. **Apply the Terraform Plan:**
    ```bash
    terraform apply -var-file="terraform.tfvars"
    ```
    Type `yes` when prompted to approve the plan.

5. **Access the Instances:**
    - For KVM:
      ```bash
      ssh cc@<kvm_floating_ip> -i ~/.ssh/id_rsa_chameleon
      ```
    - For CHI@UC:
      ```bash
      ssh cc@<chiuc_floating_ip> -i ~/.ssh/id_rsa_chameleon
      ```

---

## **Teardown the Infrastructure**
To remove the infrastructure, run:
```bash



## File Structure

terraform destroy -var-file="terraform.tfvars"
├── main.tf
├── variables.tf
├── provider.tf
├── terraform.tfvars
├── clouds.yaml
├── outputs.tf
└── .gitignore
