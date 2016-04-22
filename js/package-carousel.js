// packages carousel text tabbing shit
function hidep() {
  //go through and find all s
  for (var i = 0; i < this.length; i++) {
    z = this[i].style.zIndex;
    // hide each p that is not on top
    if (z<0) {
      this[i].children[2].hidden = true;
    }
  }
}
//init
$('a.carousel-item').hidep();
