(function($){
  $(function(){




    $('.button-collapse').sideNav();
    $('.parallax').parallax();
    $('.carousel').carousel();
    $('.slider').slider({height:600});
    $('.dropdown-button').dropdown({hover:true});
    $('.carousel').carousel();
    $('.scrollspy').scrollSpy();



    $('#gallery').click( function( e ) {
    	e.preventDefault();
    	$.swipebox( [
    		{ href:'img/pic3.jpg', title:'My Caption' },
    		{ href:'img/pic2.jpg', title:'My Second Caption' }
    	] );
    } );



    //nouislider
    /*$("#noUiSlider").noUiSlider({
       handles: 2,
       connect: true,
       scale:[10,30],
       start:[0,60]
     }); */


     //grab id and create slider
     var slider = document.getElementById('calc-slider');

     noUiSlider.create(slider, {
       start: [450],
       step: 50,
       //connect: 'lower',
       //tooltips: true,
       range: {
         'min': 250,
         'max': 1500,
       },
      });

      //show values
      var stepSliderValueElement = document.getElementById('house-val');
      var calculatorValue =  document.getElementById('calc-value');
      var imgLaptop = $('#save1');
      var imgPlane = $('#save2');
      var imgCar = $('#save3');
      var savAll = document.getElementsByClassName('saveAll');

      //savings value show
      slider.noUiSlider.on('update', function( values, handle ) {
        //saving value calculattion
        var minfee = 11850;
        var saving;
        if (values[handle] <= 300) {
          saving = minfee;
        } else {
          saving = minfee + ((values[handle] - 300) *20);
        }
        calculatorValue.innerHTML = "$" + saving;

        // change the picture that shows depending on saving

        var imageVisible = $('.imgActive');
        /*
        function ( element, className, duration ) {
        	element.classList.add(className);
        	setTimeout(function(){
        		element.classList.remove(className);
        	}, duration);
        };
        */

        if (saving <= 12000) {
          imgPlane.slideUp();
          imgLaptop.slideDown();
        } else if (saving <= 22000) {
          imgCar.slideUp();
          imgLaptop.slideUp();
          imgPlane.slideDown();
        } else if (saving <= 40000) {
          imgPlane.slideUp();
          imgCar.slideDown();
        }
      });

      //house value show
      slider.noUiSlider.on('update', function( values, handle ) {
        stepSliderValueElement.innerHTML = "$" + (values[handle] * 1000);
      });

      /* put all images in a container, which will point to an image which changes
        as the slider changes.Have to use the "on update" listener.

        Toggle() the sillouettes png; callback slideup() of red background.
      */




    // fadeIN the walk through sections
    $('.step').css("opacity", "0");
    //$('.walk-in').animate({left:200, opacity:"1"}, 3000);


    // toasts that will appear beside steps
    var $toastStep1 = $('<span>Step 1</span>');
    var $toastStep2 = $('<span>Step 2</span>');
    var $toastStep3 = $('<span>Step 3</span>');
    var $toastStep4 = $('<span>Step 4</span>');

    // init controller
			var controller = new ScrollMagic.Controller();
      var laststep4 = $('#step4').offset();
      var laststep1 = $('#step1').offset();
    // walk through section pin
      var scene = new ScrollMagic.Scene({triggerElement: '#pinX', duration:(laststep4.top - laststep1.top)})
              .setPin("#pinX", {pushFollowers: false})
              //.addIndicators({name: "1 (duration: 300)"}) // add indicators (requires plugin)
              .addTo(controller);

              // walkthoough nav link light up
            new ScrollMagic.Scene({triggerElement: "#step1"})
                    .addTo(controller)
                    .setClassToggle("#side1", "red") // add class toggle
                    .addIndicators(); // add indicators (requires plugin)
                    //.on("enter leave", function(){
                      //Materialize.toast($toastStep1, 5000);
                    //});
            new ScrollMagic.Scene({triggerElement: "#step2"})
                    .addTo(controller)
                    .setClassToggle("#side2", "red"); // add class toggle
                    //.addIndicators() // add indicators (requires plugin)
                    //.on("enter leave", function(){
                    //  Materialize.toast($toastStep2, 5000);
                    //});
            new ScrollMagic.Scene({triggerElement: "#step3"})
                    .addTo(controller)
                    .setClassToggle("#side3", "red"); // add class toggle
                    //.addIndicators() // add indicators (requires plugin)
                    //.on("enter leave", function(){
                    //  Materialize.toast($toastStep3, 5000);
                    //});
            new ScrollMagic.Scene({triggerElement: "#step4"})
                    .addTo(controller)
                    .setClassToggle("#side4", "red"); // add class toggle
                    //.addIndicators() // add indicators (requires plugin)
                    //.on("enter leave", function(){
                    //  Materialize.toast($toastStep4, 5000);
                    //});


            // scroll to fade in the steps
            new ScrollMagic.Scene({triggerElement: "#step1"})
                    .addTo(controller)
                    .on("enter", function (e) {
          					       $('#step1').animate({left:200, opacity:"1"}, 1500);
          					});
            new ScrollMagic.Scene({triggerElement: "#step2"})
                    .addTo(controller)
                    .on("enter", function (e) {
                           $('#step2').animate({left:200, opacity:"1"}, 1500);
                    });
            new ScrollMagic.Scene({triggerElement: "#step3"})
                    .addTo(controller)
                    .on("enter", function (e) {
                           $('#step3').animate({left:200, opacity:"1"}, 1500);
                    });
            new ScrollMagic.Scene({triggerElement: "#step4"})
                    .addTo(controller)
                    .on("enter", function (e) {
                           $('#step4').animate({left:200, opacity:"1"}, 1500);
                    });



    //$('.swipebox' ).swipebox();
    //swiper for propfile
    //initialize swiper when document ready
    //var mySwiper = new Swiper ('.swiper-container');

    /* pushpin implementation
    $('#walk-wrapper').pushpin({
      top: $('.walk-start').offset().top,
      bottom: (($('.walk-stop').offset().top))
    });

    $('.step').on('click', function(){
      alert($(this).offset().top)
    });
    */


    //Photography page




  }); // end of document ready
})(jQuery); // end of jQuery name space
