$( document ).ready(function (){
  $(".button-collapse").sideNav();
  $('.carousel.carousel-slider').carousel({full_width: true, time_constant: 500});

  // carousel time function
  function runner(){
    $('.carousel').carousel('next');
    setTimeout(function(){
      runner();
    }, 6000);
  }
  // delayed runtime to allow for loading
  setTimeout(function(){
    runner();
  }, 4000);

});
