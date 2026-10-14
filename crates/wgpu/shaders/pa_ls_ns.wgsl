struct Params {
    in_width: u32, in_height: u32, layers: u32,
    in_row_stride: u32, in_plane_stride: u32,
    out_row_stride: u32, out_plane_stride: u32,
    alpha_threshold: f32,
    x_offset: u32, y_offset: u32, z_offset: u32,
    batch_width: u32,
}
var<immediate> params: Params;
@group(0) @binding(0) var<storage, read> image_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> blocks_output: array<u32>;

struct Samples { colors: array<vec4<f32>, 16>, mask: u32 }
fn load_samples(block: vec3<u32>) -> Samples {
    var s: Samples;
    for (var i=0u; i<16u; i++) {
        let x=block.x*4u+i%4u; let y=block.y*4u+i/4u;
        if x<params.in_width && y<params.in_height {
            s.colors[i]=unpack4x8unorm(image_input[block.z*params.in_plane_stride+y*params.in_row_stride+x]);
            s.mask|=1u<<i;
        }
    }
    return s;
}
fn valid_block(gid: vec3<u32>) -> bool {
    return gid.x<params.batch_width && (gid.x+params.x_offset)*4u<params.in_width &&
        (gid.y+params.y_offset)*4u<params.in_height && gid.z+params.z_offset<params.layers;
}
fn block_coord(gid: vec3<u32>) -> vec3<u32> { return gid+vec3<u32>(params.x_offset,params.y_offset,params.z_offset); }
fn block_index(block: vec3<u32>) -> u32 { return block.z*params.out_plane_stride+block.y*params.out_row_stride+block.x; }
fn lex(a: vec3<f32>,b: vec3<f32>) -> bool {
    return a.x<b.x || (a.x==b.x && (a.y<b.y || (a.y==b.y && a.z<b.z)));
}
fn unit(v: vec3<f32>) -> vec3<f32> {
    let n=length(v); if n>0.0 { return v/n; } return vec3<f32>(1,0,0);
}
fn principal_extrema(s: Samples, mask: u32) -> array<vec3<f32>,2> {
    var sorted: array<vec3<f32>,16>; var count=0u;
    for (var i=0u; i<16u; i++) { if (mask&(1u<<i))!=0u { sorted[count]=s.colors[i].rgb; count++; } }
    if count==0u { return array<vec3<f32>,2>(vec3<f32>(0),vec3<f32>(0)); }
    for (var i=1u; i<count; i++) {
        let value=sorted[i]; var j=i;
        while j>0u { if !lex(value,sorted[j-1u]) { break; } sorted[j]=sorted[j-1u]; j--; }
        sorted[j]=value;
    }
    var mean=vec3<f32>(0); var low=sorted[0]; var high=sorted[0];
    for (var i=0u; i<count; i++) { mean+=sorted[i]; low=min(low,sorted[i]); high=max(high,sorted[i]); }
    mean/=f32(count);
    var cov: array<vec3<f32>,3>;
    for (var i=0u; i<count; i++) {
        let d=sorted[i]-mean;
        for (var r=0u; r<3u; r++) { for (var c=0u; c<3u; c++) { cov[r][c]+=d[r]*d[c]; } }
    }
    for (var r=0u; r<3u; r++) { cov[r]/=f32(count); }
    let delta=high-low; let center=(low+high)*0.5;
    let diagonals=array<vec3<f32>,4>(unit(delta),unit(delta*vec3<f32>(1,1,-1)),unit(delta*vec3<f32>(1,-1,1)),unit(delta*vec3<f32>(-1,1,1)));
    var diagonal=diagonals[0]; var best_var=-1.0;
    for (var d=0u; d<4u; d++) {
        var variance=0.0;
        for (var i=0u; i<count; i++) { let t=dot(sorted[i]-center,diagonals[d]); variance+=t*t; }
        if variance>best_var { best_var=variance; diagonal=diagonals[d]; }
    }
    var axis=diagonal;
    for (var iteration=0u; iteration<10u; iteration++) {
        var next=vec3<f32>(0);
        for (var r=0u; r<3u; r++) { for (var c=0u; c<3u; c++) { next[r]+=cov[r][c]*axis[c]; } }
        let n=length(next); if n>1e-6 { axis=next/n; } else { axis=diagonal; }
    }
    var dominant=0u; if abs(axis.y)>abs(axis[dominant]) { dominant=1u; } if abs(axis.z)>abs(axis[dominant]) { dominant=2u; }
    if axis[dominant]<0.0 { axis=-axis; }
    var first=sorted[0]; var last=sorted[0]; var min_p=dot(first,axis); var max_p=min_p;
    for (var i=1u; i<count; i++) {
        let p=dot(sorted[i],axis);
        if p<min_p { first=sorted[i]; min_p=p; } if p>=max_p { last=sorted[i]; max_p=p; }
    }
    return array<vec3<f32>,2>(first,last);
}
fn quantize_rgb(v: vec3<f32>) -> u32 {
    let q=vec3<u32>(floor(clamp(v,vec3<f32>(0),vec3<f32>(1))*vec3<f32>(31,63,31)+vec3<f32>(0.5)));
    return (q.x<<11u)|(q.y<<5u)|q.z;
}
fn decode_rgb(v: u32) -> vec3<f32> { return vec3<f32>(f32((v>>11u)&31u)/31.0,f32((v>>5u)&63u)/63.0,f32(v&31u)/31.0); }
struct RgbCandidate { a: u32, b: u32, indices: u32, error: f32 }
fn evaluate_rgb(s: Samples, mask: u32, qa: u32, qb: u32, three: bool) -> RgbCandidate {
    let a=select(max(qa,qb),min(qa,qb),three); let b=select(min(qa,qb),max(qa,qb),three);
    let c0=decode_rgb(a); let c1=decode_rgb(b);
    var p=array<vec3<f32>,4>(c0,c1,fma(c1-c0,vec3<f32>(1.0/3.0),c0),fma(c0-c1,vec3<f32>(1.0/3.0),c1));
    if three { p[2]=(c0+c1)*0.5; }
    var count=select(4u,3u,three); if a==b { count=1u; }
    var result=RgbCandidate(a,b,0u,0.0);
    for (var i=0u; i<16u; i++) {
        var index=0u;
        if (mask&(1u<<i))!=0u {
            var err=1e30;
            for (var j=0u; j<count; j++) { let d=s.colors[i].rgb-p[j]; let e=dot(d,d); if e<err { err=e; index=j; } }
            result.error+=sqrt(err);
        } else if three && (s.mask&(1u<<i))!=0u { index=3u; }
        result.indices|=index<<(i*2u);
    }
    return result;
}
fn refine_rgb(s: Samples,mask: u32,initial: RgbCandidate,three: bool) -> RgbCandidate {
    var best=initial; let weights=array<f32,4>(0.0,1.0,select(1.0/3.0,0.5,three),2.0/3.0);
    for (var iteration=0u; iteration<4u; iteration++) {
        var aa=0.0; var ab=0.0; var bb=0.0; var ax=vec3<f32>(0); var bx=vec3<f32>(0);
        for (var i=0u; i<16u; i++) {
            if (mask&(1u<<i))==0u { continue; }
            let b=weights[(best.indices>>(i*2u))&3u]; let a=1.0-b;
            aa+=a*a; ab+=a*b; bb+=b*b; ax+=s.colors[i].rgb*a; bx+=s.colors[i].rgb*b;
        }
        let det=aa*bb-ab*ab; if abs(det)<1e-8 { break; }
        let next=evaluate_rgb(s,mask,quantize_rgb((ax*bb-bx*ab)/det),quantize_rgb((bx*aa-ax*ab)/det),three);
        if next.error>=best.error { break; } best=next;
    }
    return best;
}
fn encode_rgb(s: Samples,mask: u32,three: bool) -> vec2<u32> {
    let extrema=principal_extrema(s,mask);
    var best=refine_rgb(s,mask,evaluate_rgb(s,mask,quantize_rgb(extrema[0]),quantize_rgb(extrema[1]),three),three);
    let base=best;
    // Both endpoint neighborhoods use the same seed; each proposal rebuilds indices.
    for (var endpoint=0u; endpoint<2u; endpoint++) {
        let code=select(base.a,base.b,endpoint==1u);
        let q=vec3<i32>(i32((code>>11u)&31u),i32((code>>5u)&63u),i32(code&31u));
        for (var r=-1; r<=1; r++) { for (var g=-1; g<=1; g++) { for (var b=-1; b<=1; b++) {
            if r==0 && g==0 && b==0 { continue; }
            let v=q+vec3<i32>(r,g,b); if any(v<vec3<i32>(0)) || any(v>vec3<i32>(31,63,31)) { continue; }
            let proposal=(u32(v.x)<<11u)|(u32(v.y)<<5u)|u32(v.z);
            let next=evaluate_rgb(s,mask,select(proposal,base.a,endpoint==1u),select(base.b,proposal,endpoint==1u),three);
            if next.error<best.error { best=next; }
        } } }
    }
    return vec2<u32>(best.a|(best.b<<16u),best.indices);
}

struct ScalarCandidate { a: u32, b: u32, lo: u32, hi: u32, error: f32 }
fn scalar_palette(a: u32,b: u32) -> array<f32,8> {
    var p: array<f32,8>; p[0]=f32(a)/255.0; p[1]=f32(b)/255.0;
    let six=a<=b; let count=select(8u,6u,six); let denominator=select(7.0,5.0,six);
    for (var i=2u; i<count; i++) { let t=f32(i-1u)/denominator; if t<=0.5 { p[i]=fma(p[1]-p[0],t,p[0]); } else { p[i]=fma(p[0]-p[1],1.0-t,p[1]); } }
    if six { p[6]=0.0; p[7]=1.0; } return p;
}
fn scalar_index(c: ScalarCandidate,i: u32) -> u32 {
    let bit=i*3u; if bit<32u { var k=c.lo>>bit; if bit>29u { k|=c.hi<<(32u-bit); } return k&7u; } return (c.hi>>(bit-32u))&7u;
}
fn evaluate_scalar(values: array<f32,16>,mask: u32,a: u32,b: u32) -> ScalarCandidate {
    let p=scalar_palette(a,b); var result=ScalarCandidate(a,b,0u,0u,0.0);
    for (var i=0u; i<16u; i++) {
        var index=0u;
        if (mask&(1u<<i))!=0u {
            var err=1e30;
            for (var j=0u; j<8u; j++) { let e=abs(values[i]-p[j]); if e<err { err=e; index=j; } }
            result.error+=err;
        }
        let bit=i*3u; if bit<32u { result.lo|=index<<bit; if bit>29u { result.hi|=index>>(32u-bit); } } else { result.hi|=index<<(bit-32u); }
    }
    return result;
}
fn quantize_scalar(v: f32) -> u32 { return u32(floor(clamp(v,0.0,1.0)*255.0+0.5)); }
fn scalar_seed(values: array<f32,16>,mask: u32,a: f32,b: f32,six: bool) -> ScalarCandidate {
    let low=quantize_scalar(min(a,b)); let high=quantize_scalar(max(a,b));
    return evaluate_scalar(values,mask,select(high,low,six),select(low,high,six));
}
fn refine_scalar(values: array<f32,16>,mask: u32,initial: ScalarCandidate) -> ScalarCandidate {
    var best=initial;
    for (var iteration=0u; iteration<4u; iteration++) {
        var aa=0.0; var ab=0.0; var bb=0.0; var ax=0.0; var bx=0.0; let six=best.a<=best.b;
        for (var i=0u; i<16u; i++) {
            if (mask&(1u<<i))==0u { continue; }
            let k=scalar_index(best,i); if six && k>=6u { continue; }
            var b=0.0; if k==1u { b=1.0; } else if k>=2u { b=f32(k-1u)/select(7.0,5.0,six); }
            let a=1.0-b; aa+=a*a; ab+=a*b; bb+=b*b; ax+=values[i]*a; bx+=values[i]*b;
        }
        let det=aa*bb-ab*ab; if abs(det)<1e-8 { break; }
        let next=scalar_seed(values,mask,(ax*bb-bx*ab)/det,(bx*aa-ax*ab)/det,six);
        if next.error>=best.error { break; } best=next;
    }
    return best;
}
fn encode_scalar(s: Samples,channel: u32) -> vec2<u32> {
    var values: array<f32,16>;
    for (var i=0u; i<16u; i++) { values[i]=s.colors[i][channel]; }
    var low=1.0; var high=0.0;
    for (var i=0u; i<16u; i++) { if (s.mask&(1u<<i))!=0u { low=min(low,values[i]); high=max(high,values[i]); } }
    var best=refine_scalar(values,s.mask,scalar_seed(values,s.mask,low,high,false));
    let six=refine_scalar(values,s.mask,scalar_seed(values,s.mask,low,high,true)); if six.error<best.error { best=six; }
    let base=best;
    for (var endpoint=0u; endpoint<2u; endpoint++) { for (var step=-1; step<=1; step+=2) {
        var a=i32(base.a); var b=i32(base.b); if endpoint==0u { a+=step; } else { b+=step; }
        if min(a,b)<0 || max(a,b)>255 { continue; }
        let next=evaluate_scalar(values,s.mask,u32(a),u32(b)); if next.error<best.error { best=next; }
    } }
    return vec2<u32>(best.a|(best.b<<8u)|(best.lo<<16u),(best.lo>>16u)|(best.hi<<16u));
}

@compute @workgroup_size(64,1,1)
fn compress_bc1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !valid_block(gid) { return; } let block=block_coord(gid); let s=load_samples(block); let offset=block_index(block)*2u;
    var mask=s.mask;
    for (var i=0u; i<16u; i++) { if s.colors[i].a<params.alpha_threshold { mask&=~(1u<<i); } }
    var encoded: vec2<u32>;
    if mask==s.mask { encoded=encode_rgb(s,mask,false); } else { encoded=encode_rgb(s,mask,true); }
    blocks_output[offset]=encoded.x; blocks_output[offset+1u]=encoded.y;
}
@compute @workgroup_size(64,1,1)
fn compress_bc2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !valid_block(gid) { return; } let block=block_coord(gid); let s=load_samples(block); let encoded=encode_rgb(s,s.mask,false); let offset=block_index(block)*4u;
    var alpha: array<u32,2>;
    for (var i=0u; i<16u; i++) { alpha[i/8u]|=u32(floor(s.colors[i].a*15.0+0.5))<<((i%8u)*4u); }
    blocks_output[offset]=alpha[0]; blocks_output[offset+1u]=alpha[1]; blocks_output[offset+2u]=encoded.x; blocks_output[offset+3u]=encoded.y;
}
@compute @workgroup_size(64,1,1)
fn compress_bc3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !valid_block(gid) { return; } let block=block_coord(gid); let s=load_samples(block); let rgb=encode_rgb(s,s.mask,false); let alpha=encode_scalar(s,3u); let offset=block_index(block)*4u;
    blocks_output[offset]=alpha.x; blocks_output[offset+1u]=alpha.y; blocks_output[offset+2u]=rgb.x; blocks_output[offset+3u]=rgb.y;
}
@compute @workgroup_size(64,1,1)
fn compress_bc4(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !valid_block(gid) { return; } let block=block_coord(gid); let encoded=encode_scalar(load_samples(block),0u); let offset=block_index(block)*2u;
    blocks_output[offset]=encoded.x; blocks_output[offset+1u]=encoded.y;
}
@compute @workgroup_size(64,1,1)
fn compress_bc5(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !valid_block(gid) { return; } let block=block_coord(gid); let s=load_samples(block); let red=encode_scalar(s,0u); let green=encode_scalar(s,1u); let offset=block_index(block)*4u;
    blocks_output[offset]=red.x; blocks_output[offset+1u]=red.y; blocks_output[offset+2u]=green.x; blocks_output[offset+3u]=green.y;
}
