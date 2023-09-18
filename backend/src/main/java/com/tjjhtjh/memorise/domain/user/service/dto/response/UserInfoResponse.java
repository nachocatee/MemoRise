package com.tjjhtjh.memorise.domain.user.service.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class UserInfoResponse {
    Long userSeq;
    String email;
    String nickname;
    String profile;

    @Builder
    public UserInfoResponse(Long userSeq, String email, String nickname, String profile) {
        this.userSeq = userSeq;
        this.email = email;
        this.nickname = nickname;
        this.profile = profile;
    }
}
