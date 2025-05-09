# provider.tf

terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.49.0"
    }
  }
}

provider "openstack" {
  alias = "kvm"
  cloud = "kvm"
}

provider "openstack" {
  alias = "chiuc"
  cloud = "chiuc"
}
