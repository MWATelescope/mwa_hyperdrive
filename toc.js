// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="index.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded affix "><li class="part-title">Installation</li><li class="chapter-item expanded "><a href="installation/intro.html"><strong aria-hidden="true">1.</strong> How do I install hyperdrive?</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="installation/pre_compiled.html"><strong aria-hidden="true">1.1.</strong> Pre-compiled</a></li><li class="chapter-item "><a href="installation/from_source.html"><strong aria-hidden="true">1.2.</strong> From source</a></li><li class="chapter-item "><a href="installation/post.html"><strong aria-hidden="true">1.3.</strong> Post installation</a></li></ol></li><li class="chapter-item expanded "><li class="part-title">User Guide</li><li class="chapter-item expanded "><a href="user/intro.html"><strong aria-hidden="true">2.</strong> Introduction</a></li><li class="chapter-item expanded "><a href="user/help.html"><strong aria-hidden="true">3.</strong> Getting started</a></li><li class="chapter-item expanded "><a href="user/di_cal/intro.html"><strong aria-hidden="true">4.</strong> DI calibration</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="user/di_cal/tutorial.html"><strong aria-hidden="true">4.1.</strong> Tutorial</a></li><li class="chapter-item "><a href="user/di_cal/simple.html"><strong aria-hidden="true">4.2.</strong> Simple usage</a></li><li class="chapter-item "><a href="user/di_cal/out_calibrated.html"><strong aria-hidden="true">4.3.</strong> Getting calibrated data</a></li><li class="chapter-item "><div><strong aria-hidden="true">4.4.</strong> Advanced usage</div><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="user/di_cal/advanced/time_varying.html"><strong aria-hidden="true">4.4.1.</strong> Varying solutions over time</a></li></ol></li><li class="chapter-item "><a href="user/di_cal/garrawarla.html"><strong aria-hidden="true">4.5.</strong> Usage on garrawarla</a></li><li class="chapter-item "><a href="user/di_cal/how_does_it_work.html"><strong aria-hidden="true">4.6.</strong> How does it work?</a></li></ol></li><li class="chapter-item expanded "><a href="user/solutions_apply/intro.html"><strong aria-hidden="true">5.</strong> Apply solutions</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="user/solutions_apply/simple.html"><strong aria-hidden="true">5.1.</strong> Simple usage</a></li></ol></li><li class="chapter-item expanded "><a href="user/plotting.html"><strong aria-hidden="true">6.</strong> Plot solutions</a></li><li class="chapter-item expanded "><a href="user/vis_convert/intro.html"><strong aria-hidden="true">7.</strong> Convert visibilities</a></li><li class="chapter-item expanded "><a href="user/vis_simulate/intro.html"><strong aria-hidden="true">8.</strong> Simulate visibilities</a></li><li class="chapter-item expanded "><a href="user/vis_subtract/intro.html"><strong aria-hidden="true">9.</strong> Subtract visibilities</a></li><li class="chapter-item expanded "><a href="user/beam.html"><strong aria-hidden="true">10.</strong> Get beam responses</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded affix "><li class="part-title">Definitions and Concepts</li><li class="chapter-item expanded "><a href="defs/pols.html"><strong aria-hidden="true">11.</strong> Polarisations</a></li><li class="chapter-item expanded "><div><strong aria-hidden="true">12.</strong> Supported visibility formats</div><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="defs/vis_formats_read.html"><strong aria-hidden="true">12.1.</strong> Read</a></li><li class="chapter-item "><a href="defs/vis_formats_write.html"><strong aria-hidden="true">12.2.</strong> Write</a></li></ol></li><li class="chapter-item expanded "><div><strong aria-hidden="true">13.</strong> MWA-specific details</div><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="defs/mwa/metafits.html"><strong aria-hidden="true">13.1.</strong> Metafits files</a></li><li class="chapter-item "><a href="defs/mwa/delays.html"><strong aria-hidden="true">13.2.</strong> Dipole delays</a></li><li class="chapter-item "><a href="defs/mwa/dead_dipoles.html"><strong aria-hidden="true">13.3.</strong> Dead dipoles</a></li><li class="chapter-item "><a href="defs/mwa/mwaf.html"><strong aria-hidden="true">13.4.</strong> mwaf flag files</a></li><li class="chapter-item "><a href="defs/mwa/corrections.html"><strong aria-hidden="true">13.5.</strong> Raw data corrections</a></li><li class="chapter-item "><a href="defs/mwa/picket_fence.html"><strong aria-hidden="true">13.6.</strong> Picket fence obs</a></li><li class="chapter-item "><a href="defs/mwa/mwalib.html"><strong aria-hidden="true">13.7.</strong> mwalib</a></li></ol></li><li class="chapter-item expanded "><a href="defs/source_lists.html"><strong aria-hidden="true">14.</strong> Sky-model source lists</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="defs/fd_types.html"><strong aria-hidden="true">14.1.</strong> Flux-density types</a></li><li class="chapter-item "><a href="defs/source_list_hyperdrive.html"><strong aria-hidden="true">14.2.</strong> hyperdrive format</a></li><li class="chapter-item "><a href="defs/source_list_ao.html"><strong aria-hidden="true">14.3.</strong> André Offringa (ao) format</a></li><li class="chapter-item "><a href="defs/source_list_rts.html"><strong aria-hidden="true">14.4.</strong> RTS format</a></li><li class="chapter-item "><a href="defs/source_list_fits.html"><strong aria-hidden="true">14.5.</strong> FITS format</a></li></ol></li><li class="chapter-item expanded "><a href="defs/cal_sols.html"><strong aria-hidden="true">15.</strong> Calibration solutions file formats</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="defs/cal_sols_hyp.html"><strong aria-hidden="true">15.1.</strong> hyperdrive format</a></li><li class="chapter-item "><a href="defs/cal_sols_ao.html"><strong aria-hidden="true">15.2.</strong> André Offringa (ao) format</a></li><li class="chapter-item "><a href="defs/cal_sols_rts.html"><strong aria-hidden="true">15.3.</strong> RTS format</a></li></ol></li><li class="chapter-item expanded "><a href="defs/beam.html"><strong aria-hidden="true">16.</strong> Beam responses</a></li><li class="chapter-item expanded "><a href="defs/modelling/intro.html"><strong aria-hidden="true">17.</strong> Modelling visibilities</a><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="defs/modelling/rime.html"><strong aria-hidden="true">17.1.</strong> Measurement equation</a></li><li class="chapter-item "><a href="defs/modelling/estimating.html"><strong aria-hidden="true">17.2.</strong> Estimating flux densities</a></li></ol></li><li class="chapter-item expanded "><a href="defs/coords.html"><strong aria-hidden="true">18.</strong> Coordinate systems</a></li><li class="chapter-item expanded "><a href="defs/dut1.html"><strong aria-hidden="true">19.</strong> DUT1</a></li><li class="chapter-item expanded "><div><strong aria-hidden="true">20.</strong> Terminology</div><a class="toggle"><div>❱</div></a></li><li><ol class="section"><li class="chapter-item "><a href="defs/blocks.html"><strong aria-hidden="true">20.1.</strong> Timeblocks and chanblocks</a></li></ol></li><li class="chapter-item expanded "><li class="spacer"></li><li class="chapter-item expanded affix "><li class="part-title">Developer Guide</li><li class="chapter-item expanded "><a href="dev/ndarray.html"><strong aria-hidden="true">21.</strong> Multiple-dimension arrays (ndarray)</a></li><li class="chapter-item expanded "><a href="dev/vec1.html"><strong aria-hidden="true">22.</strong> Non-empty vectors (vec1)</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
