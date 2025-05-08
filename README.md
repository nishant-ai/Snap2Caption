# Snap2Caption Terraform Setup

This directory contains the Terraform setup for provisioning instances on **KVM@TACC** via OpenStack. The setup includes automatic floating IP allocation and SSH access for seamless integration with the `Snap2Caption` application.

---

## **Infrastructure Setup**
The Terraform configuration provisions the following resources:

1. **VM Instance:**
   - Image: `CC-CentOS8`
   - Flavor: `m1.medium`
   - Key Pair: `id_rsa_chameleon`

2. **Floating IP:**
   - Allocated dynamically using the `public-floating` pool.

3. **Security Groups:**
   - Default security group for SSH access.

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

3. **Apply the Terraform Plan:**
    ```bash
    terraform apply -var-file="terraform.tfvars"
    ```
    Type `yes` when prompted to approve the plan.

4. **Access the Instance:**
    ```bash
    ssh cc@<floating_ip> -i ~/.ssh/id_rsa_chameleon
    ```

---

## **Teardown the Infrastructure**
To remove the infrastructure, run:

```bash
terraform destroy -var-file="terraform.tfvars"
```

This will remove the VM instance and release the floating IP.

---

## **File Structure**
```
├── main.tf
├── variables.tf
├── provider.tf
├── terraform.tfvars
└── outputs.tf
```

---

## **Contributors**
- Harsh Golani

---

*Feel free to open issues or submit PRs for improvements.*
