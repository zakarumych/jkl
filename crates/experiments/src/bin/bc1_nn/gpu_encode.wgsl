struct Params { count: u32, mode: u32, gd_steps: u32, gd_rate: f32 }
@group(0) @binding(0) var<storage, read> rgb: array<u32>;
@group(0) @binding(1) var<storage, read_write> canonical: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<u32>>;
var<immediate> params: Params;

fn color(word: u32) -> vec3<f32> {
    return vec3<f32>(f32(word & 255u), f32((word >> 8u) & 255u), f32((word >> 16u) & 255u)) / 255.0;
}
fn lex(a: vec3<f32>, b: vec3<f32>) -> bool {
    return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}
fn unit(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len != 0.0 { return v / len; }
    return vec3<f32>(1.0, 0.0, 0.0);
}
@compute @workgroup_size(64)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    if block >= params.count { return; }
    var samples: array<vec3<f32>,16>;
    for (var i=0u; i<16u; i++) { samples[i] = color(rgb[block*16u+i]); }
    for (var i=1u; i<16u; i++) {
        let value = samples[i];
        var j=i;
        while j>0u {
            if !lex(value,samples[j-1u]) { break; }
            samples[j]=samples[j-1u]; j--;
        }
        samples[j]=value;
    }
    var mean=vec3<f32>(0.0);
    var low=samples[0]; var high=samples[0];
    for (var i=0u; i<16u; i++) { mean+=samples[i]; low=min(low,samples[i]); high=max(high,samples[i]); }
    mean/=16.0;
    var cov: array<vec3<f32>,3>;
    for (var i=0u; i<16u; i++) {
        let d=samples[i]-mean;
        for (var r=0u; r<3u; r++) { for (var c=0u; c<3u; c++) { cov[r][c]+=d[r]*d[c]; } }
    }
    for (var r=0u; r<3u; r++) { cov[r]/=16.0; }
    let delta=high-low;
    let diagonals=array<vec3<f32>,4>(unit(delta),unit(delta*vec3<f32>(1,1,-1)),unit(delta*vec3<f32>(1,-1,1)),unit(delta*vec3<f32>(-1,1,1)));
    var diagonal=vec3<f32>(0.0); var best_var=-1.0;
    let center=(high+low)*0.5;
    for (var d=0u; d<4u; d++) {
        var variance=0.0;
        for (var i=0u; i<16u; i++) { let t=dot(samples[i]-center,diagonals[d]); variance+=t*t; }
        if variance>best_var { best_var=variance; diagonal=diagonals[d]; }
    }
    var axis=diagonal;
    for (var iter=0u; iter<10u; iter++) {
        var next=vec3<f32>(0.0);
        for (var r=0u; r<3u; r++) { for (var c=0u; c<3u; c++) { next[r]+=cov[r][c]*axis[c]; } }
        let len=length(next);
        if len>1e-6 { axis=next/len; } else { axis=diagonal; }
    }
    if dot(axis,axis)<1e-12 { axis=vec3<f32>(1,0,0); }
    var dominant=0u;
    if abs(axis.y)>abs(axis[dominant]) { dominant=1u; }
    if abs(axis.z)>abs(axis[dominant]) { dominant=2u; }
    if axis[dominant]<0.0 { axis=-axis; }
    if params.mode>=11u && params.mode<=13u {
        var first=samples[0];var last=samples[0];
        var min_projection=dot(first,axis);var max_projection=min_projection;
        for(var i=1u;i<16u;i++) {
            let projection=dot(samples[i],axis);
            if projection<min_projection {first=samples[i];min_projection=projection;}
            if projection>=max_projection {last=samples[i];max_projection=projection;}
        }
        for(var c=0u;c<3u;c++) {
            canonical[block*48u+c]=first[c];canonical[block*48u+45u+c]=last[c];
        }
        return;
    }
    for (var i=1u; i<16u; i++) {
        let value=samples[i]; let projection=dot(value,axis); var j=i;
        while j>0u {
            let prev=samples[j-1u]; let p=dot(prev,axis);
            if !(projection<p || (projection==p && lex(value,prev))) { break; }
            samples[j]=prev; j--;
        }
        samples[j]=value;
    }
    for (var i=0u; i<16u; i++) { for (var c=0u; c<3u; c++) { canonical[block*48u+i*3u+c]=samples[i][c]; } }
}

fn quantize(v: vec3<f32>) -> u32 {
    let q=vec3<u32>(floor(clamp(v,vec3<f32>(0),vec3<f32>(1))*vec3<f32>(31,63,31)+vec3<f32>(0.5)));
    return (q.x<<11u)|(q.y<<5u)|q.z;
}
fn decode(v:u32) -> vec3<f32> { return vec3<f32>(f32((v>>11u)&31u)/31.0,f32((v>>5u)&63u)/63.0,f32(v&31u)/31.0); }
struct Candidate { bits: vec2<u32>, error: f32 }
fn pack(block:u32,a:vec3<f32>,b:vec3<f32>) -> Candidate {
    let qa=quantize(a); let qb=quantize(b); let q0=max(qa,qb); let q1=min(qa,qb);
    let c0=decode(q0); let c1=decode(q1);
    let first: f32=1.0/3.0;
    let second: f32=2.0/3.0;
    let palette=array<vec3<f32>,4>(c0,c1,fma(c1-c0,vec3<f32>(first),c0),fma(c0-c1,vec3<f32>(1.0-second),c1));
    var indices=0u; var error=0.0;
    var count=4u; if q0==q1 { count=1u; }
    for (var i=0u; i<16u; i++) {
        let pixel=color(rgb[block*16u+i]); var best=1e30; var index=0u;
        for (var j=0u; j<count; j++) { let e=distance(pixel,palette[j]); if e<best { best=e; index=j; } }
        indices|=index<<(i*2u); error+=best;
    }
    return Candidate(vec2<u32>(q0|(q1<<16u),indices),error);
}
fn refine(block:u32,initial:Candidate,iterations:u32) -> Candidate {
    var best=initial;
    let weights=array<f32,4>(0.0,1.0,1.0/3.0,2.0/3.0);
    for (var iteration=0u; iteration<iterations; iteration++) {
        var aa=0.0; var ab=0.0; var bb=0.0; var ax=vec3<f32>(0.0); var bx=vec3<f32>(0.0);
        for (var i=0u; i<16u; i++) {
            let b=weights[(best.bits.y>>(i*2u))&3u]; let a=1.0-b;
            aa+=a*a; ab+=a*b; bb+=b*b;
            let pixel=color(rgb[block*16u+i]); ax+=a*pixel; bx+=b*pixel;
        }
        let det=aa*bb-ab*ab;
        if abs(det)<1e-8 { break; }
        let candidate=pack(block,(bb*ax-ab*bx)/det,(aa*bx-ab*ax)/det);
        if candidate.error<best.error { best=candidate; } else { break; }
    }
    return best;
}
struct Relaxation { error:f32, ga:vec3<f32>, gb:vec3<f32> }
fn relaxed(block:u32,a:vec3<f32>,b:vec3<f32>,gradient:bool) -> Relaxation {
    let weights=array<f32,4>(0.0,1.0,1.0/3.0,2.0/3.0);
    var error=0.0;var ga=vec3<f32>(0);var gb=vec3<f32>(0);
    var count=4u;if all(a==b) {count=1u;}
    for(var i=0u;i<16u;i++) {
        let pixel=color(rgb[block*16u+i]);var best=1e30;var delta=vec3<f32>(0);var beta=0.0;
        for(var j=0u;j<count;j++) {
            let d=a*(1.0-weights[j])+b*weights[j]-pixel;let e=dot(d,d);
            if e<best {best=e;delta=d;beta=weights[j];}
        }
        let norm=sqrt(best+1e-12);error+=(norm-1e-6)/16.0;
        if gradient {let g=delta/(16.0*norm);ga+=(1.0-beta)*g;gb+=beta*g;}
    }
    return Relaxation(error,ga,gb);
}
fn relaxed_fixed(block:u32,a:vec3<f32>,b:vec3<f32>,indices:u32,gradient:bool) -> Relaxation {
    let weights=array<f32,4>(0.0,1.0,1.0/3.0,2.0/3.0);
    var error=0.0;var ga=vec3<f32>(0);var gb=vec3<f32>(0);
    for(var i=0u;i<16u;i++) {
        let beta=weights[(indices>>(i*2u))&3u];
        let delta=a*(1.0-beta)+b*beta-color(rgb[block*16u+i]);
        let norm=sqrt(dot(delta,delta)+1e-12);error+=(norm-1e-6)/16.0;
        if gradient {let g=delta/(16.0*norm);ga+=(1.0-beta)*g;gb+=beta*g;}
    }
    return Relaxation(error,ga,gb);
}
fn gradient_objective(block:u32,a:vec3<f32>,b:vec3<f32>,indices:u32,fixed:bool,gradient:bool) -> Relaxation {
    if fixed {return relaxed_fixed(block,a,b,indices,gradient);}
    return relaxed(block,a,b,gradient);
}
fn gradient_refine(block:u32,initial:Candidate,fixed_indices:bool) -> Candidate {
    var best=initial;
    var a=decode(initial.bits.x&65535u);var b=decode(initial.bits.x>>16u);
    for(var iteration=0u;iteration<params.gd_steps;iteration++) {
        let current=gradient_objective(block,a,b,initial.bits.y,fixed_indices,true);var rate=params.gd_rate;var accepted=false;
        for(var attempt=0u;attempt<9u;attempt++) {
            let next_a=clamp(a-rate*current.ga,vec3<f32>(0),vec3<f32>(1));
            let next_b=clamp(b-rate*current.gb,vec3<f32>(0),vec3<f32>(1));
            let next=gradient_objective(block,next_a,next_b,initial.bits.y,fixed_indices,false);
            if next.error<current.error {
                a=next_a;b=next_b;accepted=true;
                let candidate=pack(block,a,b);if candidate.error<best.error {best=candidate;}
                break;
            }
            rate*=0.5;
        }
        if !accepted {break;}
    }
    return best;
}
@compute @workgroup_size(64)
fn check_relaxed(@builtin(global_invocation_id) gid:vec3<u32>) {
    let block=gid.x;if block>=params.count {return;}
    let bits=output[block].x;
    let value=gradient_objective(block,decode(bits&65535u),decode(bits>>16u),output[block].y,params.mode==8u,true);
    canonical[block*48u]=value.error;
    for(var c=0u;c<3u;c++) {canonical[block*48u+1u+c]=value.ga[c];canonical[block*48u+4u+c]=value.gb[c];}
}
fn has_index_three(indices:u32) -> bool {
    return (indices&(indices>>1u)&0x55555555u)!=0u;
}
fn neighbor_candidate(block:u32,qa:u32,qb:u32,seed_indices:u32,reuse:bool) -> Candidate {
    let q0=max(qa,qb);let q1=min(qa,qb);let a=decode(q0);let b=decode(q1);
    var palette=array<vec3<f32>,4>(a,b,fma(b-a,vec3<f32>(1.0/3.0),a),fma(a-b,vec3<f32>(1.0-2.0/3.0),b));
    var indices=0u;var error=0.0;
    if reuse {
        indices=seed_indices;if qa<qb {indices^=0x55555555u;}
        if qa==qb {palette[2]=fma(b-a,vec3<f32>(0.5),a);palette[3]=vec3<f32>(0);}
        for(var i=0u;i<16u;i++) {error+=distance(color(rgb[block*16u+i]),palette[(indices>>(i*2u))&3u]);}
    } else {
        var count=4u;if qa==qb {count=1u;}
        for(var i=0u;i<16u;i++) {let pixel=color(rgb[block*16u+i]);var best=1e30;var index=0u;
            for(var j=0u;j<count;j++) {let delta=pixel-palette[j];let squared=dot(delta,delta);if squared<best {best=squared;index=j;}}
            indices|=index<<(i*2u);error+=sqrt(best);
        }
    }
    return Candidate(vec2<u32>(q0|(q1<<16u),indices),error);
}
fn neighbor_refine(block:u32,initial:Candidate,reuse:bool) -> Candidate {
    var best=initial;
    for(var round=0u;round<params.gd_steps;round++) {
        let seed=best;let qa=seed.bits.x&65535u;let qb=seed.bits.x>>16u;
        for(var endpoint=0u;endpoint<2u;endpoint++) {
            let encoded=select(qa,qb,endpoint==1u);
            let base=vec3<i32>(i32((encoded>>11u)&31u),i32((encoded>>5u)&63u),i32(encoded&31u));
            for(var dr=-1;dr<=1;dr++) {for(var dg=-1;dg<=1;dg++) {for(var db=-1;db<=1;db++) {
                if dr==0 && dg==0 && db==0 {continue;}
                let channels=base+vec3<i32>(dr,dg,db);
                if any(channels<vec3<i32>(0)) || any(channels>vec3<i32>(31,63,31)) {continue;}
                let neighbor=(u32(channels.x)<<11u)|(u32(channels.y)<<5u)|u32(channels.z);
                let next_a=select(neighbor,qa,endpoint==1u);let next_b=select(qb,neighbor,endpoint==1u);
                if reuse && next_a==next_b && has_index_three(seed.bits.y) {continue;}
                let candidate=neighbor_candidate(block,next_a,next_b,seed.bits.y,reuse);
                if candidate.error<best.error {best=candidate;}
            }}}
        }
        if all(best.bits==seed.bits) {break;}
    }
    return best;
}
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid:vec3<u32>) {
    let block=gid.x; if block>=params.count { return; }
    let base=block*48u;
    let a=vec3<f32>(canonical[base],canonical[base+1u],canonical[base+2u]);
    let b=vec3<f32>(canonical[base+45u],canonical[base+46u],canonical[base+47u]);
    var result=pack(block,a,b);
    if params.mode<3u || params.mode==4u || (params.mode>=7u && params.mode<=10u) {
        let r=block*6u;
        let candidate=pack(block,a+vec3<f32>(residual[r],residual[r+1u],residual[r+2u]),b+vec3<f32>(residual[r+3u],residual[r+4u],residual[r+5u]));
        if params.mode==0u || candidate.error<result.error { result=candidate; }
    }
    if params.mode==2u || params.mode==3u || (params.mode>=7u && params.mode<=10u) { result=refine(block,result,2u); }
    if params.mode==11u {
        if params.gd_steps==0u {result=refine(block,result,4u);}
        else {result=refine(block,result,params.gd_steps);}
    }
    if params.mode==12u || params.mode==13u {result=refine(block,result,4u);}
    if params.mode==4u || params.mode==5u || params.mode==7u || params.mode==8u {result=gradient_refine(block,result,params.mode==8u);}
    if params.mode==9u || params.mode==10u || params.mode==12u || params.mode==13u {result=neighbor_refine(block,result,params.mode==10u || params.mode==13u);}
    output[block]=result.bits;
}
