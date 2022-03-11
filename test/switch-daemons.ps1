#https://computingforgeeks.com/how-to-run-docker-containers-on-windows-server-2019/

#Uninstall your current Docker CE.
Uninstall-Package -Name docker -ProviderName DockerMSFTProvider

#Enable Nested Virtualization if you’re running Docker Containers using Linux Virtual Machine running on Hyper-V.
Get-VM WinContainerHost | Set-VMProcessor -ExposeVirtualizationExtensions $true

#Then install the current preview build of Docker EE.
Install-Module DockerProvider
Install-Package Docker -ProviderName DockerProvider -RequiredVersion preview

#Enable LinuxKit system for running Linux containers
[Environment]::SetEnvironmentVariable("LCOW_SUPPORTED", "1", "Machine")

#Restart Docker Service after the change.
Restart-Service docker
