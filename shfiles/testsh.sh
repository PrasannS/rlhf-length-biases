contains() {
  case "$1" in
    (*"$2"*) true;;
    (*) false;;
  esac
}

testf(){
    if contains $1 "http"; then 
        REWARD=$1
    else
        REWARD="no"
    fi
    echo $REWARD
}

testf "httpyah"