#https://computingforgeeks.com/how-to-run-docker-containers-on-windows-server-2019/

#Write-Host "Uninstall your current Docker CE."
#Uninstall-Package -Name docker -ProviderName DockerMSFTProvider
#
#Write-Host "Enable Nested Virtualization if you’re running Docker Containers using Linux Virtual Machine running on Hyper-V."
#Get-VM WinContainerHost | Set-VMProcessor -ExposeVirtualizationExtensions $true
#
#Write-Host "Then install the current preview build of Docker EE."
#Install-Module DockerProvider
#Install-Package Docker -ProviderName DockerProvider -RequiredVersion preview
#
#Write-Host "Enable LinuxKit system for running Linux containers"
#[Environment]::SetEnvironmentVariable("LCOW_SUPPORTED", "1", "Machine")
#
#Write-Host "Restart Docker Service after the change."
#Restart-Service docker

#https://stackoverflow.com/questions/48066994/docker-no-matching-manifest-for-windows-amd64-in-the-manifest-list-entries
#https://gist.github.com/steinwaywhw/a4cd19cda655b8249d908261a62687f8
#https://stackoverflow.com/questions/57317141/linux-docker-ee-containers-on-windows-server-2016/57691939#57691939

Write-Host "Get latest LinuxKit image"
Invoke-WebRequest -UseBasicParsing -OutFile release.zip https://github.com/linuxkit/lcow/releases/download/v4.14.35-v0.3.9/release.zip
#Remove-Item "C:\Program Files\Linux Containers" -Force -Recurse
Expand-Archive release.zip -DestinationPath "C:\Program Files\Linux Containers\."
rm release.zip

Write-Host "Set experimental to true for config"
$FILE = "C:\ProgramData\docker\config\daemon.json"
$configfile =@"
{
    "experimental": true
}
"@

Write-Host "Enable LinuxKit system for running Linux containers"
[Environment]::SetEnvironmentVariable("LCOW_SUPPORTED", "1", "Machine")

Write-Host "Copy config file with experimental set to true"
#Copy-Item -Path .\test\daemon.json -Destination $FILE -PassThru
$configfile|Out-File -FilePath $FILE -Encoding ascii -Force

Write-Host "Check the file content"
type $FILE

Write-Host "Restart Docker Service after the change"
Restart-Service docker
