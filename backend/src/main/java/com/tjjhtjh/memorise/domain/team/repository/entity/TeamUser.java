package com.tjjhtjh.memorise.domain.team.repository.entity;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class TeamUser {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long teamUserSeq;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "team_seq")
    private Team team;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_seq")
    private User user;

    @Builder
    public TeamUser(Team team, User user) {
        this.team = team;
        this.user = user;
    }
}
