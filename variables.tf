# variables.tf

# ------------------------------
# KVM Variables
# ------------------------------
variable "kvm_image_name" {
  description = "The name of the image for KVM instance."
  type        = string
  default     = "CC-Ubuntu20.04"
}

variable "kvm_flavor_name" {
  description = "The flavor size for KVM instance."
  type        = string
  default     = "m1.small"
}

variable "kvm_network_name" {
  description = "The network to attach for KVM instance."
  type        = string
  default     = "sharednet1"
}

variable "kvm_key_name" {
  description = "The name of the SSH keypair for KVM."
  type        = string
  default     = "id_rsa_chameleon"
}

variable "kvm_auth_url" {
  description = "The authentication URL for KVM"
  type        = string
}

variable "kvm_region" {
  description = "The region for KVM"
  type        = string
}

variable "kvm_application_credential_id" {
  description = "The Application Credential ID for KVM"
  type        = string
  sensitive   = true
}

variable "kvm_application_credential_secret" {
  description = "The Application Credential Secret for KVM"
  type        = string
  sensitive   = true
}

# ------------------------------
# CHI@UC Variables
# ------------------------------
variable "chiuc_image_name" {
  description = "The name of the image for CHI@UC instance."
  type        = string
  default     = "CC-Ubuntu20.04"
}

variable "chiuc_flavor_name" {
  description = "The flavor size for CHI@UC instance."
  type        = string
  default     = "baremetal"
}

variable "chiuc_network_name" {
  description = "The network to attach for CHI@UC instance."
  type        = string
  default     = "sharednet1"
}

variable "chiuc_key_name" {
  description = "The name of the SSH keypair for CHI@UC."
  type        = string
  default     = "id_rsa_chameleon"
}

variable "chiuc_auth_url" {
  description = "The authentication URL for CHI@UC"
  type        = string
}

variable "chiuc_region" {
  description = "The region for CHI@UC"
  type        = string
}

variable "chiuc_application_credential_id" {
  description = "The Application Credential ID for CHI@UC"
  type        = string
  sensitive   = true
}

variable "chiuc_application_credential_secret" {
  description = "The Application Credential Secret for CHI@UC"
  type        = string
  sensitive   = true
}
