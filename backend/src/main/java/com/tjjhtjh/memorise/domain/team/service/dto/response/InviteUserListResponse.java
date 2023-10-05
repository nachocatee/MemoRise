package com.tjjhtjh.memorise.domain.team.service.dto.response;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
public class InviteUserListResponse {

        private Long userSeq;
        private String nickname;
        private String email;
        private String profile;
        private boolean isInvited;

        public InviteUserListResponse(User user, boolean isInvited) {
            this.userSeq = user.getUserSeq();
            this.nickname = user.getNickname();
            this.email = user.getEmail();
            this.profile = user.getProfile();
            this.isInvited = isInvited;
        }
}
