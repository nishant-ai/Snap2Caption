output "instance_ip" {
  description = "The public IP address of the VM."
  value       = openstack_networking_floatingip_v2.fip.address
}
