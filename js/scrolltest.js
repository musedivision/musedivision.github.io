// scroll test using scroll magic

<div id="pin1" class="box2 blue">
			<p>Stay where you are (at least for a while).</p>
			<a href="#" class="viewsource">view source</a>
		</div>

<script>
			$(function () { // wait for document ready
				// build scene
				var scene = new ScrollMagic.Scene({triggerElement: "#trigger1", duration: 300})
								.setPin("#pin1")
								//.addIndicators({name: "1 (duration: 300)"}) // add indicators (requires plugin)
								.addTo(controller);
			});
		</script>
