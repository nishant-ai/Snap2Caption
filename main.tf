resource "openstack_compute_instance_v2" "vm_instance" {
  name            = "mlops-vm"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  key_pair        = var.key_name
  security_groups = ["default"]

  network {
    name = var.network_name
  }
}

resource "openstack_networking_floatingip_v2" "fip" {
  pool = "public"
}

resource "openstack_compute_floatingip_associate_v2" "fip_assoc" {
  floating_ip = openstack_networking_floatingip_v2.fip.address
  instance_id = openstack_compute_instance_v2.vm_instance.id
}