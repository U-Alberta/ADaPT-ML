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

Write-Host "Set Path in variable"
$FILE = "C:\ProgramData\docker\config\daemon.json"

#If the file does not exist, create it.
if (-not(Test-Path -Path $FILE -PathType Leaf)) {
     try {
         $null = New-Item -ItemType File -Path $FILE -Force -ErrorAction Stop
         Write-Host "The file [$FILE] has been created."
     }
     catch {
         throw $_.Exception.Message
     }
 }
# If the file already exists, show the message and do nothing.
 else {
     Write-Host "Cannot create [$FILE] because a file with that name already exists."
 }

Write-Host "Replace experimental: false to experimental: true"
(Get-Content -Path $FILE) -replace '"experimental": false','"experimental": true' | Set-Content -Path $FILE

Write-Host "Check the file content"
type $FILE

Write-Host "Restart Docker Service after the change"
Restart-Service docker
