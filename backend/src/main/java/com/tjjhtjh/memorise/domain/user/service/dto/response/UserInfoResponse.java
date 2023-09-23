package com.tjjhtjh.memorise.domain.user.service.dto.response;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
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

    public UserInfoResponse(User user) {
        this.userSeq = user.getUserSeq();
        this.email = user.getEmail();
        this.nickname = user.getNickname();
        this.profile = user.getProfile();
    }
}
