document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.alert').forEach(a=>{setTimeout(()=>{a.style.opacity='0';a.style.transform='translateY(-5px)';a.style.transition='.35s';setTimeout(()=>a.remove(),350)},4500)});
  document.querySelectorAll('[data-confirm]').forEach(el=>el.addEventListener('click',e=>{if(!confirm(el.dataset.confirm))e.preventDefault()}));
});
