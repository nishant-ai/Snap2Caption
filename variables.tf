variable "image_name" {
  description = "The name of the image to use for the instance."
  type        = string
  default     = "CC-Ubuntu20.04"
}

variable "flavor_name" {
  description = "The flavor size to use for the instance."
  type        = string
  default     = "m1.small"
}

variable "network_name" {
  description = "The network to attach to."
  type        = string
  default     = "sharednet1"
}

variable "key_name" {
  description = "The name of the SSH keypair to use."
  type        = string
  default     = "id_rsa_chameleon"
}
