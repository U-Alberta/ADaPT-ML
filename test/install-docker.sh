#!/bin/bash -e

mkdir -p ~/.docker/machine/cache
curl -Lo ~/.docker/machine/cache/boot2docker.iso https://github.com/boot2docker/boot2docker/releases/download/v19.03.12/boot2docker.iso
brew install docker docker-machine
docker-machine create --driver virtualbox default
docker-machine env default
eval "$(docker-machine env default)"

#        brew install docker docker-machine
#        brew cask install virtualbox
#        docker-machine create --driver virtualbox default
#        docker-machine env default
#        eval "$(docker-machine env default)"

#        HOMEBREW_NO_AUTO_UPDATE=1
#        brew install --cask docker
#        sudo /Applications/Docker.app/Contents/MacOS/Docker --unattended --install-privileged-components
#        open -a /Applications/Docker.app --args --unattended --accept-license
#        echo "We are waiting for Docker to be up and running. It can take over 2 minutes..."
#        while ! /Applications/Docker.app/Contents/Resources/bin/docker info &>/dev/null; do sleep 1; done

#  brew cask install docker
#  # allow the app to run without confirmation
#  xattr -d -r com.apple.quarantine /Applications/Docker.app
#
#  # preemptively do docker.app's setup to avoid any gui prompts
#  sudo /bin/cp /Applications/Docker.app/Contents/Library/LaunchServices/com.docker.vmnetd /Library/PrivilegedHelperTools
#  sudo /bin/cp /Applications/Docker.app/Contents/Resources/com.docker.vmnetd.plist /Library/LaunchDaemons/
#  sudo /bin/chmod 544 /Library/PrivilegedHelperTools/com.docker.vmnetd
#  sudo /bin/chmod 644 /Library/LaunchDaemons/com.docker.vmnetd.plist
#  sudo /bin/launchctl load /Library/LaunchDaemons/com.docker.vmnetd.plist
#  open -g -a Docker.app
#
# Wait for the server to start up, if applicable.
i=0
while ! docker system info &>/dev/null; do
  (( i++ == 0 )) && printf %s '-- Waiting for Docker to finish starting up...' || printf '.'
  sleep 1
done
(( i )) && printf '\n'

echo "-- Docker is ready."