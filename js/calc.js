$(function(){

    //commission calculater
    var $slide = $('#test5');
    var $answer = $('#answer')
    //change in slider value

    // change inner html of answer to new value

    $slide.on('input', function(){
      var slidevalue = $slide.val()

      $answer.html("$" +

        Math.round(
          (11850 +((($slide.val())-300000) * 0.02)))

      );
    });

}); // end of document ready

// integrating noUiSlider and upfill sillouettes

// on updates slider move background fill up
// new macbook ($3000); trip to europe 4 family ($7500); new car ($15000)
